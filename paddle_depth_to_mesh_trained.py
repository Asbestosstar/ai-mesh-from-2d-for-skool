import cv2
import torch
import numpy as np
import open3d as o3d
import os
import argparse
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from peft import PeftModel


# ==========================
# ARGUMENTS
# ==========================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from image using fine-tuned Depth Anything V2 (LoRA)"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output mesh path (.obj, .ply, etc)"
    )

    parser.add_argument(
        "--model",
        default="./depth_anything_lora",
        help="Path to fine-tuned LoRA model folder"
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera FOV in degrees"
    )

    parser.add_argument(
        "--z-scale",
        type=float,
        default=3.5,
        help="Depth scaling factor"
    )

    parser.add_argument(
        "--mask-erode",
        type=int,
        default=2,
        help="Mask erosion iterations"
    )

    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=8,
        help="Poisson reconstruction depth"
    )

    return parser.parse_args()


# ==========================
# LOAD MODEL
# ==========================

def load_model(model_path, device):

    print("[INFO] Loading fine-tuned Depth Anything V2...")

    processor = AutoImageProcessor.from_pretrained(model_path)

    base_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    )

    model = PeftModel.from_pretrained(base_model, model_path)

    # Merge LoRA weights into base model for faster inference
    model = model.merge_and_unload()

    model = model.to(device)

    if device == "cuda":
        model = model.half()

    model.eval()

    print("[INFO] Model loaded successfully")

    return processor, model


# ==========================
# DEPTH ESTIMATION
# ==========================

def estimate_depth(processor, model, image, device):

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    if device == "cuda":
        inputs = {k: v.half() for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        depth = outputs.predicted_depth.squeeze().float().cpu().numpy()

    # Normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    depth = np.sqrt(depth)

    return depth


# ==========================
# MAIN
# ==========================

def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = load_model(args.model, device)

    # ==========================
    # LOAD IMAGE
    # ==========================

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    img_bgr = cv2.imread(args.input)

    if img_bgr is None:
        raise RuntimeError("Failed to load image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    orig_h, orig_w = img_rgb.shape[:2]

    print(f"[INFO] Image size: {orig_w}x{orig_h}")

    pil_image = Image.fromarray(img_rgb)

    # ==========================
    # DEPTH INFERENCE
    # ==========================

    print("[INFO] Running depth estimation...")

    depth = estimate_depth(processor, model, pil_image, device)

    # CRITICAL FIX: resize depth to match original image resolution
    depth = cv2.resize(
     depth,
     (orig_w, orig_h),
     interpolation=cv2.INTER_CUBIC
     )

    # ==========================
    # FOREGROUND MASK
    # ==========================

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    bg_mask = np.zeros_like(gray, dtype=bool)

    seeds = [
        (0,0),
        (0,orig_w-1),
        (orig_h-1,0),
        (orig_h-1,orig_w-1)
    ]

    for sy, sx in seeds:

        if gray[sy, sx] > 180:

            mask_seed = np.zeros_like(gray, dtype=np.uint8)

            cv2.floodFill(mask_seed, None, (sx, sy), 1, loDiff=30, upDiff=30)

            bg_mask |= mask_seed.astype(bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    bg_mask = cv2.morphologyEx(
        bg_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    ).astype(bool)

    fg_mask = ~bg_mask

    if fg_mask.sum() < 0.03 * orig_h * orig_w:

        print("[WARN] Using RGB fallback mask")

        fg_mask = ~np.all(img_rgb > 230, axis=2)

        fg_mask = cv2.morphologyEx(
            fg_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        ).astype(bool)

    fg_mask = cv2.erode(
        fg_mask.astype(np.uint8),
        np.ones((4,4), np.uint8),
        iterations=args.mask_erode
    ).astype(bool)

    # ==========================
    # BACKPROJECT TO 3D
    # ==========================

    focal_length = 0.5 * orig_w / np.tan(np.radians(args.fov)/2)

    cx, cy = orig_w/2, orig_h/2

    y, x = np.mgrid[0:orig_h, 0:orig_w].astype(np.float32)

    valid = fg_mask.ravel()

    x_valid = x.ravel()[valid]
    y_valid = y.ravel()[valid]
    z_vals = depth.ravel()[valid]

    colors = img_rgb.reshape(-1,3)[valid] / 255.0

    X = (x_valid - cx) * z_vals / focal_length
    Y = (y_valid - cy) * z_vals / focal_length
    Z = z_vals * args.z_scale

    points = np.stack([X, -Y, Z], axis=1)

    # ==========================
    # CREATE POINT CLOUD
    # ==========================

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03,
            max_nn=50
        )
    )

    pcd.orient_normals_towards_camera_location([0,0,0])

    # ==========================
    # POISSON MESH
    # ==========================

    print("[INFO] Running Poisson reconstruction...")

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.poisson_depth,
        scale=1.0,
        linear_fit=False
    )

    densities = np.asarray(densities)

    mesh.remove_vertices_by_mask(
        densities < np.quantile(densities, 0.1)
    )

    mesh.compute_vertex_normals()

    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=3,
        lambda_filter=0.3
    )

    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=20000
    )

    # ==========================
    # SAVE
    # ==========================

    o3d.io.write_triangle_mesh(
        args.output,
        mesh,
        write_vertex_colors=True
    )

    print("\n✅ Mesh saved:", args.output)
    print("Vertices:", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))


# ==========================

if __name__ == "__main__":
    main()

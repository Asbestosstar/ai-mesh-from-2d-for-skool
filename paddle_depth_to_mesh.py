import cv2
import torch
import numpy as np
import open3d as o3d
import os
import argparse
from PIL import Image
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from image using Depth Anything V2"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output mesh (.obj, .ply, etc)"
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Camera field of view in degrees (default: 60)"
    )

    parser.add_argument(
        "--z-scale",
        type=float,
        default=3.5,
        help="Depth scaling factor (default: 3.5)"
    )

    parser.add_argument(
        "--mask-erode",
        type=int,
        default=2,
        help="Foreground mask erosion iterations (default: 2)"
    )

    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=8,
        help="Poisson reconstruction depth (default: 8)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    IMG_PATH = args.input
    OUTPUT_MESH = args.output
    FOV_DEG = args.fov
    Z_SCALE = args.z_scale
    MASK_ERODE_ITERATIONS = args.mask_erode
    POISSON_DEPTH = args.poisson_depth

    # LOAD DEPTH ANYTHING V2
    print("[INFO] Loading Depth Anything V2...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Base-hf",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # LOAD IMAGE
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(IMG_PATH)

    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise ValueError("Invalid image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    print(f"[INFO] Image: {orig_w}x{orig_h}")

    # INFERENCE
    print("[INFO] Running Depth Anything V2...")
    pil_image = Image.fromarray(img_rgb)
    result = pipe(pil_image)
    depth = np.array(result["depth"])

    # Normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth = np.sqrt(depth)

    # FOREGROUND MASKING
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bg_mask = np.zeros_like(gray, dtype=bool)

    seeds = [
        (0, 0),
        (0, orig_w - 1),
        (orig_h - 1, 0),
        (orig_h - 1, orig_w - 1)
    ]

    for sy, sx in seeds:
        if gray[sy, sx] > 180:
            mask_seed = np.zeros_like(gray, dtype=np.uint8)
            cv2.floodFill(mask_seed, None, (sx, sy), 1, loDiff=30, upDiff=30)
            bg_mask |= mask_seed.astype(bool)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bg_mask = cv2.morphologyEx(
        bg_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    ).astype(bool)

    fg_mask = ~bg_mask

    if fg_mask.sum() < 0.03 * orig_h * orig_w:
        print("[WARN] Low foreground — using RGB fallback")
        fg_mask = ~np.all(img_rgb > 230, axis=2)
        fg_mask = cv2.morphologyEx(
            fg_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        ).astype(bool)

    fg_mask = cv2.erode(
        fg_mask.astype(np.uint8),
        np.ones((4, 4), dtype=np.uint8),
        iterations=MASK_ERODE_ITERATIONS
    ).astype(bool)

    # BACK PROJECT
    focal_length = 0.5 * orig_w / np.tan(np.radians(FOV_DEG) / 2)
    cx, cy = orig_w / 2.0, orig_h / 2.0

    y, x = np.mgrid[0:orig_h, 0:orig_w].astype(np.float32)

    x_flat = x.ravel()
    y_flat = y.ravel()
    depth_flat = depth.ravel()
    img_flat = img_rgb.reshape(-1, 3)

    valid = fg_mask.ravel()

    if valid.sum() == 0:
        raise RuntimeError("No valid foreground pixels")

    x_valid = x_flat[valid]
    y_valid = y_flat[valid]
    z_vals = depth_flat[valid]
    colors_valid = img_flat[valid] / 255.0

    X = (x_valid - cx) * z_vals / focal_length
    Y = (y_valid - cy) * z_vals / focal_length
    Z = z_vals * Z_SCALE

    points = np.stack([X, -Y, Z], axis=1)

    # POINT CLOUD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03,
            max_nn=50
        )
    )

    pcd.orient_normals_towards_camera_location([0, 0, 0])

    # POISSON
    print(f"[INFO] Running screened Poisson (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=POISSON_DEPTH,
        scale=1.0,
        linear_fit=False,
        n_threads=1
    )

    densities = np.asarray(densities)

    if len(densities) > 0:
        mesh.remove_vertices_by_mask(
            densities < np.quantile(densities, 0.10)
        )

    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    mesh = mesh.filter_smooth_laplacian(
        number_of_iterations=3,
        lambda_filter=0.3
    )

    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=20000
    )

    # SAVE
    o3d.io.write_triangle_mesh(
        OUTPUT_MESH,
        mesh,
        write_vertex_colors=True
    )

    print(f"\n✅ Mesh saved: {OUTPUT_MESH}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")


if __name__ == "__main__":
    main()

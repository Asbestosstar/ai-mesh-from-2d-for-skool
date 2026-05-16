import bpy
import os
import math
import random
import mathutils

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(SCRIPT_DIR, "train_input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "train_output")

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
DEPTH_DIR = os.path.join(OUTPUT_DIR, "depth")

VIEWS_PER_OBJECT = 100
RESOLUTION = 512

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

# ============================================================
# EXTREME SPEED RENDER SETUP
# ============================================================

def configure_fast_render():

    scene = bpy.context.scene

    scene.render.engine = "BLENDER_EEVEE"

    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION

    scene.eevee.taa_render_samples = 1
    scene.eevee.taa_samples = 1

    scene.render.use_motion_blur = False
    scene.render.use_simplify = True

    scene.render.film_transparent = True

    bpy.context.view_layer.use_pass_z = True

    print("Fast render configured")

# ============================================================
# CLEAR SCENE FAST
# ============================================================

def clear_scene():

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

# ============================================================
# CAMERA
# ============================================================

def create_camera():

    cam_data = bpy.data.cameras.new("Camera")

    cam = bpy.data.objects.new("Camera", cam_data)

    bpy.context.collection.objects.link(cam)

    bpy.context.scene.camera = cam

    cam.data.clip_start = 0.01
    cam.data.clip_end = 100

    return cam

# ============================================================
# CAMERA POSITION
# ============================================================

def random_camera(radius=2.0):

    theta = random.random() * 2 * math.pi

    phi = random.uniform(
        math.radians(15),
        math.radians(75)
    )

    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi)
    )

# ============================================================
# LOOK AT TARGET
# ============================================================

def look_at(camera):

    direction = mathutils.Vector((0,0,0)) - camera.location

    camera.rotation_euler = direction.to_track_quat(
        '-Z','Y'
    ).to_euler()

# ============================================================
# IMPORT OBJ FAST
# ============================================================

def import_obj(path):

    bpy.ops.wm.obj_import(filepath=path)

    return bpy.context.selected_objects[0]

# ============================================================
# NORMALIZE OBJECT FAST
# ============================================================

def normalize_object(obj):

    max_dim = max(obj.dimensions)

    if max_dim == 0:
        max_dim = 1

    scale = 1.0 / max_dim

    obj.scale = (scale, scale, scale)
    obj.location = (0,0,0)

# ============================================================
# SAVE DEPTH FAST
# ============================================================

def save_depth(name, index):

    path = os.path.join(
        DEPTH_DIR,
        f"{name}_{index:04d}.exr"
    )

    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"

    bpy.data.images["Render Result"].save_render(path)

# ============================================================
# SAVE RGB FAST
# ============================================================

def save_rgb(path):

    bpy.context.scene.render.image_settings.file_format = "PNG"

    bpy.context.scene.render.filepath = path

# ============================================================
# RENDER OBJECT FAST
# ============================================================

def render_object(path, camera):

    obj = import_obj(path)

    normalize_object(obj)

    name = os.path.splitext(
        os.path.basename(path)
    )[0]

    print("Rendering:", name)

    for i in range(VIEWS_PER_OBJECT):

        camera.location = random_camera()

        look_at(camera)

        rgb_path = os.path.join(
            IMAGE_DIR,
            f"{name}_{i:04d}.png"
        )

        save_rgb(rgb_path)

        bpy.ops.render.render(write_still=True)

        save_depth(name, i)

    bpy.ops.object.delete()

# ============================================================
# MAIN
# ============================================================

def main():

    clear_scene()

    configure_fast_render()

    camera = create_camera()

    objs = [
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".obj")
    ]

    print("Found", len(objs), "objects")

    for obj in objs:

        render_object(
            os.path.join(INPUT_DIR, obj),
            camera
        )

    print("DATASET COMPLETE")

# ============================================================

if __name__ == "__main__":
    main()

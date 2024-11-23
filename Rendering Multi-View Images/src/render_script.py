import bpy
import os
from mathutils import Vector


def load_3Dmodel():

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "car.blend")

    # Clear the current Blender scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Link or append objects from the Blend file
    with bpy.data.libraries.load(model_path, link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]

    # Add imported objects to the scene
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)

def combine_objects():

    # Get all mesh objects
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not imported_objects:
        print("No mesh objects found. Ensure the model is correctly imported!")
        return None

    # Apply transformations to all objects
    for obj in imported_objects:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Select all objects
    for obj in imported_objects:
        obj.select_set(True)

    # Join all objects into a single object
    bpy.context.view_layer.objects.active = imported_objects[0]
    bpy.ops.object.join()

    # Get the combined object
    combined_object = bpy.context.object
    combined_object.name = "Combined_Object"

    return combined_object

def move_bbox_to_origin(obj):

    # Get the bounding box vertices of the object
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    # Calculate the center of the bounding box
    bbox_center = sum(bbox, Vector()) / 8

    # Move the object
    obj.location -= bbox_center

def setup_lights():

    center_light_data = bpy.data.lights.new(name="Center_Light", type='POINT')
    center_light = bpy.data.objects.new(name="Center_Light", object_data=center_light_data)
    bpy.context.collection.objects.link(center_light)
    center_light.location = (0, 0, 2)
    center_light.data.energy = 200

    diagonal_positions = [
        (-2, -2, 2),
        (2, -2, 2),
        (-2, 2, 2),
        (2, 2, 2)
    ]

    for i, pos in enumerate(diagonal_positions):
        light_data = bpy.data.lights.new(name=f"Diagonal_Light_{i+1}", type='POINT')
        light_object = bpy.data.objects.new(name=f"Diagonal_Light_{i+1}", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = pos
        light_object.data.energy = 150

def render_views(output_dir):

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up the camera
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object

    # Define rendering angles and positions
    views = {
        "front": ((5, 0, 0), (1.5708, 0, 1.5708)),    # Front
        "back": ((-5, 0, 0), (1.5708, 0, -1.5708)),  # Back
        "left": ((0, 8, 0), (-1.5708, 3.14159, 0)),   # Left
        "right": ((0, -8, 0), (1.5708, 0, 0))        # Right
    }

    for view, (location, rotation) in views.items():
        camera_object.location = location
        camera_object.rotation_euler = rotation

        # Set render output file
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"{view}.png")
        bpy.ops.render.render(write_still=True)
        print(f"{view} view rendered and saved to {bpy.context.scene.render.filepath}")


def save_combined_object():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    output_file = os.path.join(output_dir, "car_final.blend")

    # Save the current scene as a .blend file
    bpy.ops.wm.save_as_mainfile(filepath=output_file)


if __name__ == "__main__":
    # Load the model from the Blend file
    load_3Dmodel()

    # Combine all mesh objects
    combined_object = combine_objects()

    # Print result
    if combined_object:

        move_bbox_to_origin(combined_object)

        setup_lights()

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        render_views(output_dir)

        save_combined_object()
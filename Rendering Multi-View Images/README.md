# 3D Model Processing and Rendering

This project provides a Python script to process and render 3D models using Blender. 

## Features

- Load 3D models from `.blend` files.
- Normalize the model and centering
- Add custom lighting to the scene.
- Render views from predefined angles.

## How to Run

1. **Install Blender**:
   Ensure Blender is installed on your system and accessible from the command line.

2. **Install Python Dependencies**:
   Install the required Python libraries using `pip`:
   ```bash
   pip install -r requirements.txt

3. **Run the Script: Use the following command to run the script with Blender:**
   ```bash
   cd your/path/src
   blender --background --python render_script.py
   
5. **Check Outputs:**
Rendered images and Combined models are saved in the output directory.

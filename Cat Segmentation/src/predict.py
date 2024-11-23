import os
import torch
from PIL import Image
import numpy as np
from UNet import UNet
from Dataset import get_transform

def load_model(model_path):
    """Load the trained model."""
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully!")
    return model

def predict_single(image_path, model, transform):
    """Predict a single image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        mask = output.squeeze(0).squeeze(0).numpy()  # Remove batch and channel dims
        mask = (mask > 0.5).astype(np.uint8)  # Binarize the mask

    return np.array(image), mask

def predict_folder(input_folder, output_folder, model, transform):
    """
    Predict all images in the input folder and save results to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(input_folder, filename)
        image, mask = predict_single(input_path, model, transform)

        # Save the predicted mask
        mask_image = Image.fromarray((mask * 255).astype("uint8"))
        mask_image.save(os.path.join(output_folder, f"segmented_{filename}"))

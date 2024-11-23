import os
from PIL import Image
from predict import load_model, predict_single, predict_folder
from Dataset import get_transform
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model_path = "results/model.pth"
    test_input_dir = "data/Test_Data"
    output_dir = "data/Result_Data"

    print("Loading model for prediction...")
    model = load_model(model_path)

    transform = get_transform()

    print("Predicting and saving results...")
    predict_folder(test_input_dir, output_dir, model, transform)

    print("Visualizing some results...")
    for idx, filename in enumerate(os.listdir(test_input_dir)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        input_path = os.path.join(test_input_dir, filename)
        original_image, predicted_mask = predict_single(input_path, model, transform)

        # Visualize first few results
        if idx < 5:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(original_image)

            plt.subplot(1, 2, 2)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask, cmap="gray")

            plt.show()

import torch
from Train import train_model
from Dataset import get_transform
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_input_dir = "data/input"
    train_mask_dir = "data/mask"
    model_path = "results/model.pth"

    epochs = 200
    batch_size = 8
    lr = 0.0001

    print("Training model...")
    model, dataloader = train_model(
        image_dir=train_input_dir,
        mask_dir=train_mask_dir,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        return_dataloader=True
    )

    print("Visualizing training results...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for idx, (images, masks) in enumerate(dataloader):
        if idx >= 1:
            break
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)

        for i in range(min(len(images), 5)):
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(images[i].cpu().permute(1, 2, 0))

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Mask")
            plt.imshow(masks[i].cpu().squeeze(), cmap="gray")

            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(outputs[i].cpu().squeeze(), cmap="gray")

            plt.show()

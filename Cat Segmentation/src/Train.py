import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Dataset import CatSegmentationDataset, get_transform
from UNet import UNet
from losses import DiceBCELoss

def train_model(image_dir, mask_dir, model_path, epochs=50, batch_size=8, lr=1e-4, return_dataloader=False):
    transform = get_transform()
    dataset = CatSegmentationDataset(image_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if return_dataloader:
        return model, dataloader
    return model

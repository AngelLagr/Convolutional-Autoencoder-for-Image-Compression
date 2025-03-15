import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from convAutoencoder import ConvAutoencoder

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load and preprocess the dataset
print("🔄 Loading and transforming the dataset...")
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to a fixed size
    transforms.ToTensor()            # Convert images to PyTorch tensors
])

dataset = ImageFolder(root="./dataset", transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Dataset loaded successfully! {len(dataset)} images available \n")

# Initialize the model, loss function, and optimizer
print("🔄 Initializing model, loss function, and optimizer...\n")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
print("🔄 Model ready! Training will start soon...\n")

# Training loop
num_epochs = 2
print("Training started")
for epoch in range(num_epochs):
    total_loss = 0.0  # Track loss for the epoch
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)  # Move images to GPU if available
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), "autoencoder.pth")

    # Print epoch summary
    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch [{epoch+1}/{num_epochs}] completed! Average Loss: {avg_loss:.4f}\n")

print("Training finished! Model saved as autoencoder.pth")
torch.save(model.state_dict(), "autoencoder.pth")
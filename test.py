import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from convAutoencoder import ConvAutoencoder
from PIL import Image
import matplotlib.pyplot as plt

# Part needded to stop the libiomp5md.dll conflict (on my setup)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check if CUDA is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} \n")



# Load and preprocess the dataset
print("🔄 Loading and transforming the dataset...")
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to a fixed size
    transforms.ToTensor()            # Convert images to PyTorch tensors
])

dataset = ImageFolder(root="./dataset", transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Dataset loaded successfully! {len(dataset)} images available\n")

# Get a batch of test images
images, _ = next(iter(dataloader))
images = images.to(device)

# Load the model
model = ConvAutoencoder()
model.load_state_dict(torch.load("autoencoder.pth"))
model.to(device)
model.eval()  # avoid training error

print("Model loaded successfully \n")

# Pass through autoencoder
with torch.no_grad():
    reconstructions = model(images)
    
# Convert to CPU for visualization
images = images.cpu().numpy().transpose(0, 2, 3, 1) 
reconstructions = reconstructions.cpu().numpy().transpose(0, 2, 3, 1)

# Plot original vs reconstructed images
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for i in range(6):
    axes[0, i].imshow(images[i])  
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructions[i])  
    axes[1, i].axis('off')
axes[0, 0].set_title("Original Images")
axes[1, 0].set_title("Reconstructed Images")

plt.show()
print("Reconstructed images plotted\n")

# Load the test
image_path = "./bigimage.jpg"
original_image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),        
])
input_image = transform(original_image).unsqueeze(0)  

# Seek the original size of the picture
original_size = os.path.getsize(image_path) / 1024  # Size in Ko
print(f"Original Image Size: {original_size:.2f} KB")

input_image = input_image.to(device)

# Encode the picture
with torch.no_grad():
    encoded = model.encoder(input_image)


encoded_size = encoded.element_size() * encoded.nelement() / 1024  # size in Ko
print(f"Encoded Picture Size: {encoded_size:.2f} KB \n")

# Compare the sizes
compression_ratio = original_size / encoded_size
print(f"Compression Ratio: {compression_ratio:.2f}x")


# Reconstruct the encoded picture
with torch.no_grad():
    reconstructed_image = model.decoder(encoded)  # Reconstruct the picture

# Prepare the plot
original_image = original_image.resize((512, 512)) 
reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)  

# Plot the difference
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[0].axis('off')
ax[0].set_title("Original Image")

ax[1].imshow(reconstructed_image)
ax[1].axis('off')
ax[1].set_title("Reconstructed Image")

plt.show()

# Convolutional-Autoencoder-for-Image-Compression
This project focuses on image compression using a convolutional autoencoder. The goal is to compress satellite images from the EuroSAT dataset into a compact representation and reconstruct them with minimal quality loss.

The convolutional autoencoder model is trained to learn efficient image representations. It uses a mean squared error (MSE) loss function to minimize the difference between the original and the reconstructed images. The model is implemented using PyTorch and trained using the Adam optimizer.

The model is trained on EuroSAT images, resized to 64x64 pixels, and evaluated on the ability to reconstruct the images after compression.


---

## Project Structure  
- **`train.py`**: Script for training the convolutional autoencoder on the EuroSAT dataset.  
- **`test.py`**: Script to test the model on images from the dataset.  
- **`convAutoencoder.py`**: Defines the architecture of the convolutional autoencoder model.  
- **`autoencoder.pth`**: File storing the trained model weights.  
- **`results.png`**: Visual comparison between original and compressed images (to be generated after running the model).  

---

## Technologies Used  
- **Python**  
- **PyTorch** (for building and training the convolutional autoencoder)  
- **EuroSAT dataset** (satellite imagery)  
- **Matplotlib** (for result visualization)  

---

## Execution  
To train the model:  
```bash
python train.py
```

To test the model on images:
```bash
python test.py
```

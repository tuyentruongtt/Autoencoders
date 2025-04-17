import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

class ParticlePatchCAE(nn.Module):
    def __init__(self, input_channels=3, input_size=64, latent_dim=64, num_classes=2):
        super(ParticlePatchCAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        class_logits = self.classifier(features)
        return reconstruction, class_logits, features


class ParticleAnnotationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, patch_size=64):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.patch_size = patch_size
        self.samples = []

        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(image_dir, filename)
                base_name = os.path.splitext(filename)[0]
                txt_path = os.path.join(label_dir, base_name + ".txt")
                if not os.path.exists(txt_path):
                    continue
                W, H = 512, 512
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        cls = int(parts[0])
                        x_center = float(parts[1]) * W
                        y_center = float(parts[2]) * H
                        self.samples.append((img_path, x_center, y_center, cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, x_center, y_center, cls = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        x1 = int(x_center - self.patch_size // 2)
        y1 = int(y_center - self.patch_size // 2)
        x1 = max(0, min(x1, 512 - self.patch_size))
        y1 = max(0, min(y1, 512 - self.patch_size))

        patch = img.crop((x1, y1, x1 + self.patch_size, y1 + self.patch_size))

        if self.transform:
            patch = self.transform(patch)

        return patch, cls


def visualize_latent_space(features, labels, save_path='tsne_latent.png'):
    features = features.view(features.size(0), -1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE of Latent Space')
    plt.savefig(save_path)
    plt.close()
    
def visualize_reconstructions(model, dataloader, num_samples=5, save_path='reconstructions.png'):
    """
    Visualize original images and their reconstructions side by side
    
    Args:
        model: Trained ParticlePatchCAE model
        dataloader: DataLoader with image data
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    plt.figure(figsize=(12, 2 * num_samples))
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            reconstructions, _, _ = model(images)
            
            # Get the first image from the batch
            orig_img = images[0].cpu().numpy().transpose(1, 2, 0)
            recon_img = reconstructions[0].cpu().numpy().transpose(1, 2, 0)
            
            # Plot original
            plt.subplot(num_samples, 2, i*2 + 1)
            plt.imshow(orig_img)
            plt.title(f"Original (Class {labels[0].item()})")
            plt.axis('off')
            
            # Plot reconstruction
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.imshow(recon_img)
            plt.title(f"Reconstruction")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Reconstructions saved to {save_path}")

def visualize_by_class(model, dataloader, save_path='class_reconstructions.png'):
    """
    Create a grid of reconstructions organized by class
    """
    model.eval()
    class_samples = {0: [], 1: []}  # Assuming 2 classes
    
    # Collect samples by class
    with torch.no_grad():
        for images, labels in dataloader:
            for i in range(len(images)):
                img = images[i:i+1]
                label = labels[i].item()
                
                if len(class_samples[label]) < 5:  # Get 5 samples of each class
                    recon, _, features = model(img)
                    
                    # Store original, reconstruction and its latent features
                    class_samples[label].append({
                        'original': img[0].cpu().numpy().transpose(1, 2, 0),
                        'reconstruction': recon[0].cpu().numpy().transpose(1, 2, 0),
                        'features': features[0].cpu().numpy()
                    })
    
    # Create visualization
    num_classes = len(class_samples)
    samples_per_class = min([len(samples) for samples in class_samples.values()])
    
    plt.figure(figsize=(10, 2 * num_classes * samples_per_class))
    
    for class_idx in range(num_classes):
        for i in range(samples_per_class):
            sample = class_samples[class_idx][i]
            
            # Original
            plt.subplot(num_classes * samples_per_class, 2, (class_idx * samples_per_class + i) * 2 + 1)
            plt.imshow(sample['original'])
            plt.title(f"Class {class_idx} - Original")
            plt.axis('off')
            
            # Reconstruction
            plt.subplot(num_classes * samples_per_class, 2, (class_idx * samples_per_class + i) * 2 + 2)
            plt.imshow(sample['reconstruction'])
            plt.title(f"Class {class_idx} - Reconstruction")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Class-based reconstructions saved to {save_path}")

def train_patch_cae():
    image_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/train/images'
    label_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/train/labels'

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = ParticleAnnotationDataset(image_dir, label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ParticlePatchCAE()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    all_losses = []
    all_features = []
    all_labels = []

    for epoch in range(10):
        total_loss = 0
        for images, labels in dataloader:
            recon, logits, features = model(images)
            loss_recon = criterion_recon(recon, images)
            loss_class = criterion_class(logits, labels)
            loss = loss_recon + 1.0 * loss_class

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_losses.append(loss.item())

            all_features.append(features)
            all_labels.append(labels)

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), 'particle_patch_cae.pth')

    plt.figure()
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    visualize_latent_space(all_features, all_labels)


if __name__ == "__main__":
    train_patch_cae()

def evaluate_model():
    # Define the transform again
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    # Load the trained model
    model = ParticlePatchCAE()
    model.load_state_dict(torch.load('particle_patch_cae.pth'))
    model.eval()
    
    # Create a test dataloader with a small batch size
    test_image_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/test/images'
    test_label_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/test/labels'
    
    test_dataset = ParticleAnnotationDataset(test_image_dir, test_label_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Visualize reconstructions
    visualize_reconstructions(model, test_dataloader, num_samples=5, save_path='particle_reconstructions.png')
    
    # You can also compute quantitative metrics here
    total_loss = 0
    criterion_recon = nn.MSELoss()
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            recon, logits, _ = model(images)
            loss = criterion_recon(recon, images)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_dataloader)
    print(f"Average reconstruction loss on test set: {avg_loss:.4f}")

# Call this after training is complete
evaluate_model()
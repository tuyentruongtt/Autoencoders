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


def visualize_latent_space_balanced(model, dataloader, save_path='tsne_latent.png', 
                              max_samples=2000, balanced=True):
    """
    Perform t-SNE visualization on dataset samples with memory optimization
    and optional class balancing
    
    Args:
        model: Trained model
        dataloader: DataLoader with image data
        save_path: Path to save the visualization
        max_samples: Maximum number of samples to use for t-SNE
        balanced: If True, try to balance classes in visualization
    """
    model.eval()
    
    if balanced:
        # Approach 1: Collect balanced samples per class
        class_samples = defaultdict(list)
        class_counts = defaultdict(int)
        
        # First pass: count total samples per class
        total_classes = 0
        with torch.no_grad():
            for images, labels in dataloader:
                for label in labels.cpu().numpy():
                    class_counts[label] += 1
                total_classes = max(total_classes, labels.max().item() + 1)
        
        # Calculate samples per class to balance the visualization
        if len(class_counts) > 0:
            # Aim for balanced representation but limit total samples
            samples_per_class = min(
                min(class_counts.values()),  # Don't exceed available samples
                max_samples // len(class_counts)  # Don't exceed max_samples
            )
            
            # Second pass: collect balanced samples
            with torch.no_grad():
                for images, labels in dataloader:
                    # Get features
                    _, _, batch_features = model(images)
                    batch_features = batch_features.view(batch_features.size(0), -1)
                    
                    # Add features by class
                    for i, label in enumerate(labels):
                        label_idx = label.item()
                        if len(class_samples[label_idx]) < samples_per_class:
                            class_samples[label_idx].append({
                                'features': batch_features[i].cpu().numpy(),
                                'label': label_idx
                            })
            
            # Combine samples from all classes
            features_list = []
            labels_list = []
            
            for class_idx in range(total_classes):
                samples = class_samples[class_idx]
                if samples:
                    for sample in samples:
                        features_list.append(sample['features'])
                        labels_list.append(sample['label'])
            
            features = np.vstack(features_list)
            labels = np.array(labels_list)
            
            print(f"Created balanced t-SNE visualization with {len(features)} samples "
                  f"({samples_per_class} per class)")

    else:
        # Approach 2: Sample randomly up to max_samples
        features_list = []
        labels_list = []
        sample_count = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                # Generate features
                _, _, batch_features = model(images)
                batch_features = batch_features.view(batch_features.size(0), -1).cpu().numpy()
                batch_labels = labels.cpu().numpy()
                
                features_list.append(batch_features)
                labels_list.append(batch_labels)
                
                sample_count += len(images)
                if sample_count >= max_samples:
                    break
        
        # Combine all collected samples
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        
        # Randomly subsample if we have too many
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        print(f"Created random sampled t-SNE visualization with {len(features)} samples")

    # Apply t-SNE with appropriate perplexity
    # Perplexity should be smaller than n_samples - 1
    perplexity = min(30, len(features) - 1)
    print(f"Running t-SNE on {len(features)} samples with perplexity {perplexity}...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    embedded = tsne.fit_transform(features)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', 
                          alpha=0.7, s=10)  # s=point size
    
    # Add color bar and potential class labels
    cbar = plt.colorbar(scatter, label='Class')
    
    # If we have reasonably few classes, add class labels to colorbar
    if total_classes <= 10:
        cbar.set_ticks(np.arange(total_classes))
        cbar.set_ticklabels([f'Class {i}' for i in range(total_classes)])
    
    plt.title('t-SNE Visualization of Latent Space')
    plt.savefig(save_path, dpi=300)  # Higher DPI for better quality
    plt.close()
    
    print(f"t-SNE visualization saved to {save_path}")
    
    # Return stats for reporting
    return {
        'num_samples': len(features),
        'num_classes': len(np.unique(labels)),
        'samples_per_class': {i: np.sum(labels == i) for i in np.unique(labels)}
    }
    
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

# Helper functions
def load_dataset(image_dir, label_dir, image_size=(64, 64), batch_size=32, shuffle=True):
    """Helper function to load the dataset."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    dataset = ParticleAnnotationDataset(image_dir, label_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset

def load_trained_model(path='particle_patch_cae.pth'):
    """Helper function to load a trained model."""
    model = ParticlePatchCAE()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Main training function
def train_patch_cae():
    image_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/train/images'
    label_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/train/labels'

    # Use the helper function to load the dataset
    dataloader, dataset = load_dataset(image_dir, label_dir)

    # Initialize model and optimizer
    model = ParticlePatchCAE()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    all_losses = []

    # Training loop
    for epoch in range(10):
        total_loss = 0
        for images, labels in dataloader:
            recon, logits, _ = model(images)
            loss_recon = criterion_recon(recon, images)
            loss_class = criterion_class(logits, labels)
            loss = loss_recon + 1.0 * loss_class

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_losses.append(loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'particle_patch_cae.pth')

    # Plot training loss
    plt.figure()
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()

    # Switch to eval mode and re-extract features for t-SNE
    model.eval()

    # Reload dataloader with batch_size=1 for better t-SNE performance
    tsne_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    try:
        tsne_stats = visualize_latent_space_balanced(
            model,
            tsne_dataloader,
            save_path='tsne_latent_balanced_train.png',
            max_samples=2000,
            balanced=True
        )
        print(f"Training t-SNE stats: {tsne_stats}")
    except Exception as e:
        print(f"Failed to generate t-SNE on training data: {e}")
    
# Main evaluation function
def evaluate_model():
    # Set test data paths
    image_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/test/images'
    label_dir = '/home/jupyter-tuyen/PatchCAE/PatchCAE_images/Dataset/test/labels'

    # Use the helper function to load test dataset
    dataloader, dataset = load_dataset(image_dir, label_dir)

    # Load trained model
    model = load_trained_model()
    
    # Visualize reconstructions using a small batch size
    viz_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    visualize_reconstructions(model, viz_dataloader, num_samples=5, save_path='particle_reconstructions.png')

    # Compute average reconstruction loss on test set
    criterion_recon = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for images, _ in dataloader:
            recon, _, _ = model(images)
            loss = criterion_recon(recon, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average reconstruction loss on test set: {avg_loss:.4f}")
    
    # Generate balanced t-SNE visualization
    try:
        tsne_stats = visualize_latent_space_balanced(
            model, 
            dataloader,
            save_path='tsne_latent_balanced.png',
            max_samples=2000,
            balanced=True
        )
        print(f"t-SNE visualization stats: {tsne_stats}")
    except Exception as e:
        print(f"t-SNE visualization failed: {e}")

if __name__ == "__main__":
    train_patch_cae()
    evaluate_model()
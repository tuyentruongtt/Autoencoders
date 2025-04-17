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

class ParticlePatchCAE(nn.Module):
    def __init__(self, input_channels=3, input_size=64, latent_dim=64, num_classes=2):
        super(ParticlePatchCAE, self).__init__()

        self.feature_size = input_size // 16

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


def train_patch_cae():
    image_dir = 'path_to_rgb_images/'
    label_dir = 'path_to_label_txts/'

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
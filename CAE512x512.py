import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import seaborn as sns
from typing import List, Tuple, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
import seaborn as sns
   
class ParticleCAE(nn.Module):
    def __init__(self, input_size=512, latent_dim=64):
        """
        Convolutional Autoencoder for particle analysis
        Args:
            input_size: Size of input images (assumes square)
            latent_dim: Dimension of latent space
        """
        super(ParticleCAE, self).__init__()

        self.feature_size = input_size // 32

        # Encoder
        self.encoder = nn.Sequential(
            # Level 1: 512x512 -> 256x256
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Level 2: 256x256 -> 128x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Level 3: 128x128 -> 64x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Level 4: 64x64 -> 32x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Level 5: 32x32 -> 16x16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Classification branch
        self.class_branch = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

        # Position estimation branch (modify this in your ParticleCAE class)
        self.position_branch = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=1)  # Changed from 4 to 2 for (x,y) coordinates
        )

        # Spread estimation branch
        self.spread_branch = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1)  # σx and σy for two classes
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Level 5: 16x16 -> 32x32
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),

            # Level 4: 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),

            # Level 3: 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            # Level 2: 128x128 -> 256x256
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            # Level 1: 256x256 -> 512x512
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input validation
        assert x.size(2) == 512 and x.size(3) == 512, f"Input size must be 512x512, got {x.size(2)}x{x.size(3)}"
        assert x.size(1) == 1, f"Input must be grayscale (1 channel), got {x.size(1)} channels"

        # Encode
        features = self.encoder(x)

        # Get classifications
        class_logits = self.class_branch(features).squeeze(-1).squeeze(-1)

        # Get position estimates (x,y coordinates only)
        positions = self.position_branch(features)
        positions = positions.mean(dim=(2, 3))  # Output shape: [batch_size, 2]

        # Get spread estimates
        spreads = self.spread_branch(features)
        spreads = spreads.mean(dim=(2, 3))

        # Decode
        reconstruction = self.decoder(features)

        return reconstruction, class_logits, positions, spreads
    

class ParticleDataset(Dataset):
    """Dataset class for particle images"""
    def __init__(self, image_paths: List[str], labels: Dict = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            # Properly handle the labels
            return img, {
                'classes': self.labels['classes'][idx],
                'positions': self.labels['positions'][idx],
                'spreads': self.labels['spreads'][idx]
            }
        return img

class ParticleDistributionLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_pos=1.0, lambda_spread=1.0):
        super(ParticleDistributionLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_pos = lambda_pos
        self.lambda_spread = lambda_spread

    def forward(self, pred, target, return_components=False):
        reconstruction, class_logits, positions, spreads = pred
        original_image, target_dict = target
        
        # Extract targets from dictionary
        true_classes = target_dict['classes']  # Shape: [batch_size, n_particles]
        true_positions = target_dict['positions']  # Shape: [batch_size, n_particles, 2]
        true_spreads = target_dict['spreads']  # Shape: [batch_size, n_particles]
        
        # Reconstruction loss
        rec_loss = F.mse_loss(reconstruction, original_image)
        
        # Class loss - use first particle in each image
        first_class = true_classes[:, 0]
        class_loss = F.cross_entropy(class_logits, first_class)
        
        # Position loss - use position of first particle
        first_pos = true_positions[:, 0, :]  # Shape: [batch_size, 2]
        pos_loss = F.mse_loss(positions, first_pos)
        
        # Spread loss - use spread of first particle
        first_spread = true_spreads[:, 0].unsqueeze(1)  # Shape: [batch_size, 1]
        spread_values = spreads[:, :2]  # Take first 2 values (σx and σy for first class)
        spread_mean = spread_values.mean(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        spread_loss = F.mse_loss(spread_mean, first_spread)

        total_loss = (self.lambda_rec * rec_loss +
                     class_loss +
                     self.lambda_pos * pos_loss +
                     self.lambda_spread * spread_loss)
        
        if return_components:
            return total_loss, {
                'rec_loss': rec_loss.item(),
                'class_loss': class_loss.item(),
                'pos_loss': pos_loss.item(),
                'spread_loss': spread_loss.item()
            }
        
        return total_loss

def create_data_loader(
    image_paths: List[str],
    batch_size: int,
    labels: Dict = None,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for batch processing"""
    dataset = ParticleDataset(image_paths, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def prepare_training_data(
    image_dir: str,
    label_file: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders"""
    # Load image paths
    image_paths = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
    ]

    # Load labels
    labels_data = np.load(label_file, allow_pickle=True)
    
    # Extract data and convert to appropriate formats
    classes = torch.tensor(np.array(labels_data['classes'].tolist()), dtype=torch.long)
    positions = torch.tensor(np.array(labels_data['positions'].tolist()), dtype=torch.float32)
    spreads = torch.tensor(np.array(labels_data['spreads'].tolist()), dtype=torch.float32)
    
    # Print shapes for debugging
    print(f"Classes shape: {classes.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Spreads shape: {spreads.shape}")
    
    # Create label dictionary
    label_dict = {
        'classes': classes,
        'positions': positions,
        'spreads': spreads
    }

    # Split data
    n_samples = len(image_paths)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    # Create train dataset
    train_paths = [image_paths[i] for i in train_indices]
    train_labels = {
        'classes': label_dict['classes'][train_indices],
        'positions': label_dict['positions'][train_indices],
        'spreads': label_dict['spreads'][train_indices]
    }

    # Create validation dataset
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = {
        'classes': label_dict['classes'][val_indices],
        'positions': label_dict['positions'][val_indices],
        'spreads': label_dict['spreads'][val_indices]
    }

    return create_data_loader(train_paths, batch_size, train_labels, num_workers, True), \
           create_data_loader(val_paths, batch_size, val_labels, num_workers, False)

# Modified training function with early stopping removed
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    device: torch.device = None,
    save_path: str = None
) -> Dict:
    """Train the particle analysis model with detailed metrics tracking"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ParticleDistributionLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Enhanced history dictionary to track detailed metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rec_loss': [],
        'val_rec_loss': [],
        'train_class_loss': [],
        'val_class_loss': [],
        'train_pos_loss': [],
        'val_pos_loss': [],
        'train_spread_loss': [],
        'val_spread_loss': [],
        'train_class_acc': [],
        'val_class_acc': [],
        'train_pos_error': [],
        'val_pos_error': [],
        'train_spread_error': [],
        'val_spread_error': [],
        'best_val_loss': float('inf'),
        'learning_rates': []
    }

    # Remove the early stopping setup
    # patience = 10
    # early_stopping_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_metrics = defaultdict(float)
        train_batches = 0
        
        # Track correct classifications and total samples
        train_correct = 0
        train_total = 0
        
        # Track position and spread errors
        train_pos_errors = []
        train_spread_errors = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            # Get batch data
            images = batch[0].to(device)
            labels = {
                'classes': batch[1]['classes'].to(device),
                'positions': batch[1]['positions'].to(device),
                'spreads': batch[1]['spreads'].to(device)
            }

            # Forward pass
            outputs = model(images)
            reconstruction, class_logits, positions, spreads = outputs
            
            # Compute loss with component breakdown
            loss, loss_components = criterion(outputs, (images, labels), return_components=True)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            for k, v in loss_components.items():
                train_metrics[k] += v
                
            train_metrics['total_loss'] += loss.item()
            train_batches += 1
            
            # Classification accuracy
            _, predicted = torch.max(class_logits, 1)
            true_classes = labels['classes'][:, 0]  # First particle in each image
            train_correct += (predicted == true_classes).sum().item()
            train_total += true_classes.size(0)
            
            # Position error (euclidean distance)
            true_positions = labels['positions'][:, 0, :]  # Shape: [batch_size, 2]
            pos_error = torch.sqrt(((positions - true_positions) ** 2).sum(dim=1))
            train_pos_errors.extend(pos_error.cpu().detach().numpy())
            
            # Spread error
            true_spreads = labels['spreads'][:, 0].unsqueeze(1)
            spread_values = spreads[:, :2].mean(dim=1, keepdim=True)
            spread_error = torch.abs(spread_values - true_spreads)
            train_spread_errors.extend(spread_error.cpu().detach().numpy())
            
            pbar.set_postfix({'train_loss': f'{train_metrics["total_loss"]/train_batches:.4f}'})

        # Calculate average training metrics
        for k in train_metrics:
            train_metrics[k] /= train_batches
        
        train_class_acc = 100 * train_correct / train_total if train_total > 0 else 0
        train_avg_pos_error = np.mean(train_pos_errors)
        train_avg_spread_error = np.mean(train_spread_errors)
        
        # Update history
        history['train_loss'].append(train_metrics['total_loss'])
        history['train_rec_loss'].append(train_metrics['rec_loss'])
        history['train_class_loss'].append(train_metrics['class_loss'])
        history['train_pos_loss'].append(train_metrics['pos_loss'])
        history['train_spread_loss'].append(train_metrics['spread_loss'])
        history['train_class_acc'].append(train_class_acc)
        history['train_pos_error'].append(train_avg_pos_error)
        history['train_spread_error'].append(train_avg_spread_error)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_metrics = defaultdict(float)
            val_batches = 0
            
            # Track correct classifications and total samples
            val_correct = 0
            val_total = 0
            
            # Track position and spread errors
            val_pos_errors = []
            val_spread_errors = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(device)
                    labels = {
                        'classes': batch[1]['classes'].to(device),
                        'positions': batch[1]['positions'].to(device),
                        'spreads': batch[1]['spreads'].to(device)
                    }

                    outputs = model(images)
                    reconstruction, class_logits, positions, spreads = outputs
                    
                    # Compute loss with component breakdown
                    loss, loss_components = criterion(outputs, (images, labels), return_components=True)

                    # Update metrics
                    for k, v in loss_components.items():
                        val_metrics[k] += v
                        
                    val_metrics['total_loss'] += loss.item()
                    val_batches += 1
                    
                    # Classification accuracy
                    _, predicted = torch.max(class_logits, 1)
                    true_classes = labels['classes'][:, 0]  # First particle in each image
                    val_correct += (predicted == true_classes).sum().item()
                    val_total += true_classes.size(0)
                    
                    # Position error (euclidean distance)
                    true_positions = labels['positions'][:, 0, :]  # Shape: [batch_size, 2]
                    pos_error = torch.sqrt(((positions - true_positions) ** 2).sum(dim=1))
                    val_pos_errors.extend(pos_error.cpu().detach().numpy())
                    
                    # Spread error
                    true_spreads = labels['spreads'][:, 0].unsqueeze(1)
                    spread_values = spreads[:, :2].mean(dim=1, keepdim=True)
                    spread_error = torch.abs(spread_values - true_spreads)
                    val_spread_errors.extend(spread_error.cpu().detach().numpy())

            # Calculate average validation metrics
            for k in val_metrics:
                val_metrics[k] /= val_batches
            
            val_class_acc = 100 * val_correct / val_total if val_total > 0 else 0
            val_avg_pos_error = np.mean(val_pos_errors)
            val_avg_spread_error = np.mean(val_spread_errors)
            
            # Update history
            history['val_loss'].append(val_metrics['total_loss'])
            history['val_rec_loss'].append(val_metrics['rec_loss'])
            history['val_class_loss'].append(val_metrics['class_loss'])
            history['val_pos_loss'].append(val_metrics['pos_loss'])
            history['val_spread_loss'].append(val_metrics['spread_loss'])
            history['val_class_acc'].append(val_class_acc)
            history['val_pos_error'].append(val_avg_pos_error)
            history['val_spread_error'].append(val_avg_spread_error)

            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])

            # Save best model (still keep this part for model checkpointing)
            if val_metrics['total_loss'] < history['best_val_loss']:
                history['best_val_loss'] = val_metrics['total_loss']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history,
                        'val_metrics': val_metrics,
                    }, save_path)
                # Remove early stopping counter reset
                # early_stopping_counter = 0
            # else:
                # Remove early stopping counter increment
                # early_stopping_counter += 1

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training - Loss: {train_metrics["total_loss"]:.4f}, Rec: {train_metrics["rec_loss"]:.4f}, ' 
                  f'Class Acc: {train_class_acc:.2f}%, Pos Err: {train_avg_pos_error:.4f}')
            print(f'Validation - Loss: {val_metrics["total_loss"]:.4f}, Rec: {val_metrics["rec_loss"]:.4f}, '
                  f'Class Acc: {val_class_acc:.2f}%, Pos Err: {val_avg_pos_error:.4f}')

            # Remove early stopping check
            # if early_stopping_counter >= patience:
            #     print(f'Early stopping triggered after {epoch+1} epochs')
            #     break
        else:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training - Loss: {train_metrics["total_loss"]:.4f}, Rec: {train_metrics["rec_loss"]:.4f}, ' 
                  f'Class Acc: {train_class_acc:.2f}%, Pos Err: {train_avg_pos_error:.4f}')

    return history

# Function to plot training metrics
def plot_training_metrics(history):
    """Plot detailed training metrics from history"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot Component Losses
    axes[0, 1].plot(epochs, history['train_rec_loss'], 'b-', label='Train Reconstruction')
    if 'val_rec_loss' in history and history['val_rec_loss']:
        axes[0, 1].plot(epochs, history['val_rec_loss'], 'r-', label='Val Reconstruction')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Classification Loss and Accuracy
    axes[1, 0].plot(epochs, history['train_class_loss'], 'b-', label='Train Classification Loss')
    if 'val_class_loss' in history and history['val_class_loss']:
        axes[1, 0].plot(epochs, history['val_class_loss'], 'r-', label='Val Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, history['train_class_acc'], 'b-', label='Train Accuracy')
    if 'val_class_acc' in history and history['val_class_acc']:
        axes[1, 1].plot(epochs, history['val_class_acc'], 'r-', label='Val Accuracy')
    axes[1, 1].set_title('Classification Accuracy')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot Position and Spread Errors
    axes[2, 0].plot(epochs, history['train_pos_error'], 'b-', label='Train Position Error')
    if 'val_pos_error' in history and history['val_pos_error']:
        axes[2, 0].plot(epochs, history['val_pos_error'], 'r-', label='Val Position Error')
    axes[2, 0].set_title('Position Prediction Error')
    axes[2, 0].set_xlabel('Epochs')
    axes[2, 0].set_ylabel('Error (pixels)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(epochs, history['train_spread_error'], 'b-', label='Train Spread Error')
    if 'val_spread_error' in history and history['val_spread_error']:
        axes[2, 1].plot(epochs, history['val_spread_error'], 'r-', label='Val Spread Error')
    axes[2, 1].set_title('Spread Prediction Error')
    axes[2, 1].set_xlabel('Epochs')
    axes[2, 1].set_ylabel('Error')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    return fig

# Enhanced model evaluation function
def evaluate_model_detailed(model, data_loader, device=None):
    """
    Perform detailed evaluation of the model performance
    Returns comprehensive metrics and visualizations
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    criterion = ParticleDistributionLoss()
    
    # Metric containers
    metrics = {
        'total_loss': 0.0,
        'rec_loss': 0.0,
        'class_loss': 0.0,
        'pos_loss': 0.0,
        'spread_loss': 0.0,
        'class_correct': 0,
        'class_total': 0,
        'position_errors': [],
        'spread_errors': [],
        'mse_values': [],
        'ssim_values': []
    }
    
    num_batches = 0
    all_predictions = []
    all_true_values = []
    
    try:
        from skimage.metrics import structural_similarity
        ssim_available = True
    except ImportError:
        ssim_available = False
        print("Warning: skimage not available, SSIM metrics will not be calculated")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch[0].to(device)
            labels = {
                'classes': batch[1]['classes'].to(device),
                'positions': batch[1]['positions'].to(device),
                'spreads': batch[1]['spreads'].to(device)
            }
            
            # Forward pass
            outputs = model(images)
            reconstruction, class_logits, positions, spreads = outputs
            
            # Compute loss components
            loss, loss_components = criterion(outputs, (images, labels), return_components=True)
            
            # Update metrics
            metrics['total_loss'] += loss.item()
            for k, v in loss_components.items():
                metrics[k] += v
            
            # Calculate classification accuracy
            _, predicted = torch.max(class_logits, 1)
            true_classes = labels['classes'][:, 0]  # First particle in each image
            metrics['class_correct'] += (predicted == true_classes).sum().item()
            metrics['class_total'] += true_classes.size(0)
            
            # Store predictions and true values for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_true_values.extend(true_classes.cpu().numpy())
            
            # Calculate position error
            true_positions = labels['positions'][:, 0, :]  # Shape: [batch_size, 2]
            pos_errors = torch.sqrt(((positions - true_positions) ** 2).sum(dim=1))
            metrics['position_errors'].extend(pos_errors.cpu().numpy())
            
            # Calculate spread error
            true_spreads = labels['spreads'][:, 0].unsqueeze(1)
            spread_values = spreads[:, :2].mean(dim=1, keepdim=True)
            spread_errors = torch.abs(spread_values - true_spreads)
            metrics['spread_errors'].extend(spread_errors.cpu().numpy())
            
            # Calculate image reconstruction metrics
            for i in range(images.size(0)):
                # MSE
                mse = F.mse_loss(images[i], reconstruction[i]).item()
                metrics['mse_values'].append(mse)
                
                # SSIM if available
                if ssim_available:
                    orig = images[i].squeeze().cpu().numpy()
                    recon = reconstruction[i].squeeze().cpu().numpy()
                    ssim = structural_similarity(orig, recon, data_range=1.0)
                    metrics['ssim_values'].append(ssim)
            
            num_batches += 1
    
    # Calculate aggregate metrics
    metrics['avg_total_loss'] = metrics['total_loss'] / num_batches
    metrics['avg_rec_loss'] = metrics['rec_loss'] / num_batches
    metrics['avg_class_loss'] = metrics['class_loss'] / num_batches
    metrics['avg_pos_loss'] = metrics['pos_loss'] / num_batches
    metrics['avg_spread_loss'] = metrics['spread_loss'] / num_batches
    
    metrics['classification_accuracy'] = 100 * metrics['class_correct'] / metrics['class_total']
    metrics['avg_position_error'] = np.mean(metrics['position_errors'])
    metrics['avg_spread_error'] = np.mean(metrics['spread_errors'])
    metrics['avg_mse'] = np.mean(metrics['mse_values'])
    
    if ssim_available:
        metrics['avg_ssim'] = np.mean(metrics['ssim_values'])
    
    # Print detailed evaluation results
    print(f"\nDetailed Evaluation Results:")
    print(f"Average Total Loss: {metrics['avg_total_loss']:.4f}")
    print(f"Average Reconstruction Loss: {metrics['avg_rec_loss']:.4f}")
    print(f"Average Classification Loss: {metrics['avg_class_loss']:.4f}")
    print(f"Average Position Loss: {metrics['avg_pos_loss']:.4f}")
    print(f"Average Spread Loss: {metrics['avg_spread_loss']:.4f}")
    print(f"Classification Accuracy: {metrics['classification_accuracy']:.2f}%")
    print(f"Average Position Error: {metrics['avg_position_error']:.4f} pixels")
    print(f"Average Spread Error: {metrics['avg_spread_error']:.4f}")
    print(f"Average MSE: {metrics['avg_mse']:.4f}")
    
    if ssim_available:
        print(f"Average SSIM: {metrics['avg_ssim']:.4f}")
    
    # Create confusion matrix plot if we have classifications
    if len(all_predictions) > 0:
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            import seaborn as sns
            
            # Build confusion matrix
            cm = confusion_matrix(all_true_values, all_predictions)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            
            # Generate classification report
            report = classification_report(all_true_values, all_predictions)
            print("\nClassification Report:")
            print(report)
            
        except ImportError:
            print("sklearn or seaborn not available for confusion matrix visualization")
    
    # Plot error distributions
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(metrics['mse_values'], bins=20, alpha=0.7)
    plt.axvline(metrics['avg_mse'], color='r', linestyle='--')
    plt.title('MSE Distribution')
    plt.xlabel('MSE')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    plt.hist(metrics['position_errors'], bins=20, alpha=0.7)
    plt.axvline(metrics['avg_position_error'], color='r', linestyle='--')
    plt.title('Position Error Distribution')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 3)
    plt.hist(metrics['spread_errors'], bins=20, alpha=0.7)
    plt.axvline(metrics['avg_spread_error'], color='r', linestyle='--')
    plt.title('Spread Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('error_distributions.png')
    
    return metrics

def visualize_reconstructions(model, data_loader, num_samples=5, device=None):
    """
    Visualize original images and their reconstructions from the model
    
    Args:
        model: Trained ParticleCAE model
        data_loader: DataLoader containing validation or test data
        num_samples: Number of samples to visualize
        device: Device to run inference on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Get sample batch
    samples = next(iter(data_loader))
    images = samples[0][:num_samples].to(device)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructions, _, _, _ = model(images)
    
    # Move tensors to CPU for visualization
    images = images.cpu()
    reconstructions = reconstructions.cpu()
    
    # Plot the original and reconstructed images
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Plot original image
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Plot reconstruction
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Calculate metrics
    mse_values = []
    ssim_values = []
    
    try:
        from skimage.metrics import structural_similarity
        
        for i in range(num_samples):
            # Calculate Mean Squared Error
            mse = F.mse_loss(images[i], reconstructions[i]).item()
            mse_values.append(mse)
            
            # Calculate Structural Similarity Index
            orig = images[i].squeeze().numpy()
            recon = reconstructions[i].squeeze().numpy()
            ssim = structural_similarity(orig, recon, data_range=1.0)
            ssim_values.append(ssim)
    
        # Display metrics
        print("Image Reconstruction Metrics:")
        for i in range(num_samples):
            print(f"Sample {i+1}: MSE = {mse_values[i]:.4f}, SSIM = {ssim_values[i]:.4f}")
        
        # Average metrics
        avg_mse = sum(mse_values) / len(mse_values)
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print(f"Average: MSE = {avg_mse:.4f}, SSIM = {avg_ssim:.4f}")
            
    except ImportError:
        print("skimage not available for SSIM calculation. Only showing visual comparison.")
    
    return fig

def evaluate_model(model, data_loader, device=None):
    """Evaluate the model performance on a dataset"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    criterion = ParticleDistributionLoss()
    total_loss = 0.0
    num_batches = 0
    
    classification_correct = 0
    classification_total = 0
    position_errors = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch[0].to(device)
            labels = {
                'classes': batch[1]['classes'].to(device),
                'positions': batch[1]['positions'].to(device),
                'spreads': batch[1]['spreads'].to(device)
            }
            
            # Forward pass
            outputs = model(images)
            reconstruction, class_logits, positions, spreads = outputs
            
            # Compute loss
            loss = criterion(outputs, (images, labels))
            total_loss += loss.item()
            
            # Calculate classification accuracy (for first particle)
            _, predicted = torch.max(class_logits, 1)
            true_classes = labels['classes'][:, 0]  # First particle in each image
            classification_correct += (predicted == true_classes).sum().item()
            classification_total += true_classes.size(0)
            
            # Calculate position error (for first particle)
            true_positions = labels['positions'][:, 0, :]  # Shape: [batch_size, 2]
            position_error = torch.sqrt(((positions - true_positions) ** 2).sum(dim=1)).mean().item()
            position_errors.append(position_error)
            
            num_batches += 1
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    classification_accuracy = 100 * classification_correct / classification_total
    avg_position_error = sum(position_errors) / len(position_errors)
    
    print(f"Evaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Classification Accuracy: {classification_accuracy:.2f}%")
    print(f"Average Position Error: {avg_position_error:.4f} pixels")
    
    return {
        'avg_loss': avg_loss,
        'classification_accuracy': classification_accuracy,
        'avg_position_error': avg_position_error
    }

def extract_latent_features(model, data_loader, device=None):
    """
    Extract the latent space features from the bottleneck layer of the model
    
    Args:
        model: Trained ParticleCAE model
        data_loader: DataLoader containing the data
        device: Device to run inference on
    
    Returns:
        features: Extracted latent features (N x C x H x W)
        labels: Corresponding labels
        positions: Corresponding positions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_features = []
    all_labels = []
    all_positions = []
    
    # Create a hook to capture the output of the encoder
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook on the encoder
    handle = model.encoder.register_forward_hook(get_activation('encoder'))
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            images = batch[0].to(device)
            # Get encoder output by running a forward pass
            _ = model(images)
            
            # Extract features from the activation
            features = activation['encoder']
            
            # Extract labels and positions
            labels = batch[1]['classes'][:, 0].cpu().numpy()  # First particle class
            positions = batch[1]['positions'][:, 0, :].cpu().numpy()  # First particle position
            
            # Store features, labels, and positions
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_positions.append(positions)
    
    # Remove the hook
    handle.remove()
    
    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.concatenate(all_labels)
    all_positions = np.concatenate(all_positions)
    
    return all_features, all_labels, all_positions

def visualize_latent_space_pca(features, labels, positions=None, n_components=2):
    """
    Visualize the latent space using PCA
    
    Args:
        features: Extracted latent features (N x C x H x W)
        labels: Corresponding labels
        positions: Corresponding positions
        n_components: Number of PCA components
    """
    # Reshape features to 2D (N x (C*H*W))
    features_flat = features.reshape(features.shape[0], -1).numpy()
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_flat)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot by labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('PCA of Latent Space Colored by Class')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    
    # Plot by positions if available
    if positions is not None:
        plt.subplot(1, 2, 2)
        
        # Color by position magnitude (distance from origin)
        pos_magnitude = np.sqrt(np.sum(positions**2, axis=1))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=pos_magnitude, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Position Magnitude')
        plt.title('PCA of Latent Space Colored by Position Magnitude')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_space_pca.png')
    
    return pca, features_pca

def visualize_latent_space_tsne(features, labels, positions=None, perplexity=30, 
                               sample_size=None, random_seed=42):
    """
    Visualize the latent space using t-SNE with optional subsampling
    
    Args:
        features: Extracted latent features (N x C x H x W)
        labels: Corresponding labels
        positions: Corresponding positions
        perplexity: t-SNE perplexity parameter
        sample_size: Number of samples to use (None = use all data)
        random_seed: Random seed for reproducibility
    """
    # Sample a subset if specified
    if sample_size is not None and sample_size < features.shape[0]:
        np.random.seed(random_seed)
        indices = np.random.choice(features.shape[0], sample_size, replace=False)
        features_subset = features[indices]
        labels_subset = labels[indices]
        positions_subset = positions[indices] if positions is not None else None
        print(f"Sampling {sample_size} examples from {features.shape[0]} total")
    else:
        features_subset = features
        labels_subset = labels
        positions_subset = positions
    
    # Reshape features to 2D (N x (C*H*W))
    features_flat = features_subset.reshape(features_subset.shape[0], -1).numpy()
    
    # Apply t-SNE
    from sklearn.manifold import TSNE
    print(f"Running t-SNE on {features_flat.shape[0]} samples (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, features_flat.shape[0]-1), 
               n_iter=1000, verbose=1)
    features_tsne = tsne.fit_transform(features_flat)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot by labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                         c=labels_subset, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE of Latent Space ({features_flat.shape[0]} samples) Colored by Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    # Plot by positions if available
    if positions_subset is not None:
        plt.subplot(1, 2, 2)
        # Color by position magnitude (distance from origin)
        pos_magnitude = np.sqrt(np.sum(positions_subset**2, axis=1))
        scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                             c=pos_magnitude, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Position Magnitude')
        plt.title(f't-SNE of Latent Space ({features_flat.shape[0]} samples) Colored by Position Magnitude')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_space_tsne.png')
    print(f"Visualization saved to 'latent_space_tsne.png'")
    
    return features_tsne, indices if sample_size is not None else None

def visualize_latent_space_umap(features, labels, positions=None, n_neighbors=15, min_dist=0.1):
    """
    Visualize the latent space using UMAP
    
    Args:
        features: Extracted latent features (N x C x H x W)
        labels: Corresponding labels
        positions: Corresponding positions
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    try:
        import umap
    except ImportError:
        print("UMAP is not installed. Please install it using: pip install umap-learn")
        return None
    
    # Reshape features to 2D (N x (C*H*W))
    features_flat = features.reshape(features.shape[0], -1).numpy()
    
    # Apply UMAP
    print("Running UMAP (this may take a while)...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, verbose=True)
    features_umap = reducer.fit_transform(features_flat)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot by labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('UMAP of Latent Space Colored by Class')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(True, alpha=0.3)
    
    # Plot by positions if available
    if positions is not None:
        plt.subplot(1, 2, 2)
        
        # Color by position magnitude (distance from origin)
        pos_magnitude = np.sqrt(np.sum(positions**2, axis=1))
        scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=pos_magnitude, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Position Magnitude')
        plt.title('UMAP of Latent Space Colored by Position Magnitude')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_space_umap.png')
    
    return features_umap

def visualize_feature_maps(model, data_loader, num_samples=5, max_features=16, device=None):
    """
    Visualize feature maps from the bottleneck layer of the model
    
    Args:
        model: Trained ParticleCAE model
        data_loader: DataLoader containing the data
        num_samples: Number of samples to visualize
        max_features: Maximum number of feature maps to display per sample
        device: Device to run inference on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Get a batch of data
    images, _ = next(iter(data_loader))
    images = images[:num_samples].to(device)
    
    # Create a hook to capture the output of the encoder
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook on the encoder
    handle = model.encoder.register_forward_hook(get_activation('encoder'))
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    # Get feature maps
    feature_maps = activation['encoder']
    
    # Remove the hook
    handle.remove()
    
    # Move to CPU for visualization
    feature_maps = feature_maps.cpu()
    images = images.cpu()
    
    # Visualize feature maps for each sample
    for i in range(num_samples):
        # Create a figure for this sample
        n_features = min(feature_maps.size(1), max_features)
        fig, axes = plt.subplots(1 + (n_features + 3) // 4, 4, figsize=(15, 3 * (1 + (n_features + 3) // 4)))
        
        # Plot original image
        axes[0, 0].imshow(images[i].squeeze(), cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Plot feature maps
        for j in range(1, n_features + 1):
            row, col = (j + 3) // 4, (j + 3) % 4
            if col == 0:
                col = 4
                row -= 1
            
            feature = feature_maps[i, j-1]
            
            # Normalize feature map for visualization
            vmin, vmax = feature.min(), feature.max()
            if vmax > vmin:
                feature = (feature - vmin) / (vmax - vmin)
            
            axes[row, col-1].imshow(feature, cmap='viridis')
            axes[row, col-1].set_title(f'Feature {j}')
            axes[row, col-1].axis('off')
        
        # Hide unused subplots
        for j in range(n_features + 1, (1 + (n_features + 3) // 4) * 4):
            row, col = j // 4, j % 4
            if col == 0:
                col = 4
                row -= 1
            if row < len(axes) and col-1 < len(axes[0]):
                axes[row, col-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'feature_maps_sample_{i+1}.png')
        plt.close()
    
    print(f"Feature maps visualizations saved for {num_samples} samples")

def visualize_latent_correlations(features, labels, positions):
    """
    Visualize correlations between latent space components and particle attributes
    
    Args:
        features: Extracted latent features (N x C x H x W)
        labels: Corresponding labels
        positions: Corresponding positions (N x 2)
    """
    # Reshape features to 2D (N x (C*H*W))
    features_flat = features.reshape(features.shape[0], -1).numpy()
    
    # Apply PCA to reduce dimensionality
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)  # Take top 10 components
    features_pca = pca.fit_transform(features_flat)
    
    # Create correlation dataframe
    import pandas as pd
    
    # Create a dataframe with PCA components
    df = pd.DataFrame(features_pca, columns=[f'PC{i+1}' for i in range(features_pca.shape[1])])
    
    # Add labels and positions
    df['class'] = labels
    df['pos_x'] = positions[:, 0]
    df['pos_y'] = positions[:, 1]
    df['pos_magnitude'] = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    
    # Calculate correlations
    corr = df.corr()
    
    # Visualize correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlations between Latent Components and Particle Attributes')
    plt.tight_layout()
    plt.savefig('latent_correlations.png')
    
    # Get the components most correlated with position and class
    pos_x_corr = corr['pos_x'].iloc[:10].abs().sort_values(ascending=False)
    pos_y_corr = corr['pos_y'].iloc[:10].abs().sort_values(ascending=False)
    class_corr = corr['class'].iloc[:10].abs().sort_values(ascending=False)
    
    print("\nLatent components most correlated with particle position X:")
    print(pos_x_corr)
    
    print("\nLatent components most correlated with particle position Y:")
    print(pos_y_corr)
    
    print("\nLatent components most correlated with particle class:")
    print(class_corr)
    
    return corr

# Add these function calls to your main() function
def visualize_latent_space(model, data_loader, device=None):
    """
    Perform comprehensive latent space visualization
    
    Args:
        model: Trained ParticleCAE model
        data_loader: DataLoader containing the data
        device: Device to run inference on
    """
    print("\nExtracting latent features...")
    features, labels, positions = extract_latent_features(model, data_loader, device)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of samples: {len(labels)}")
    
    # Visualize with PCA
    print("\nVisualizing latent space with PCA...")
    pca, features_pca = visualize_latent_space_pca(features, labels, positions)
    
    # Visualize with t-SNE (if dataset is not too large)
    if len(labels) <= 5000:
        print("\nVisualizing latent space with t-SNE...")
        features_tsne = visualize_latent_space_tsne(features, labels, positions)
    else:
        print(f"\nSkipping t-SNE visualization due to large dataset size ({len(labels)} samples)")
    
    # Visualize with UMAP (if installed and dataset is not too large)
    try:
        import umap
        if len(labels) <= 10000:
            print("\nVisualizing latent space with UMAP...")
            features_umap = visualize_latent_space_umap(features, labels, positions)
        else:
            print(f"\nSkipping UMAP visualization due to large dataset size ({len(labels)} samples)")
    except ImportError:
        print("\nUMAP is not installed. Skipping UMAP visualization.")
    
    # Visualize feature maps
    print("\nVisualizing feature maps...")
    visualize_feature_maps(model, data_loader, num_samples=3, max_features=16, device=device)
    
    # Visualize correlations
    print("\nAnalyzing latent space correlations...")
    corr = visualize_latent_correlations(features, labels, positions)
    
    print("\nLatent space visualizations complete.")
    
    return features, labels, positions

# Example usage
def main():
    # Setup
    image_dir = "/home/jupyter-tuyen/Autoencoders/Images/Dataset/train/images"
    label_file = "/home/jupyter-tuyen/Autoencoders/Images/Dataset/train/labels.npz"
    save_path = "best_model.pth"

    # Initialize model
    model = ParticleCAE()

    # Prepare data
    train_loader, val_loader = prepare_training_data(
        image_dir=image_dir,
        label_file=label_file,
        batch_size=32
    )

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=70,
        save_path=save_path
    )

    # Visualize reconstructions
    print("\nGenerating reconstruction visualizations...")
    visualization = visualize_reconstructions(
        model=model,
        data_loader=val_loader,
        num_samples=5
    )
    
    # Save the visualization
    visualization.savefig('reconstruction_comparison.png')
    print("Visualization saved as 'reconstruction_comparison.png'")
    
    # Also add a function to evaluate the model on test data
    print("\nEvaluating model on validation data...")
    evaluate_model(model, val_loader)
     # Plot training metrics
    plot_training_metrics(history)

    # Visualize reconstructions
    print("\nGenerating reconstruction visualizations...")
    visualization = visualize_reconstructions(
        model=model,
        data_loader=val_loader,
        num_samples=5
    )
    
    # Save the visualization
    visualization.savefig('reconstruction_comparison.png')
    print("Visualization saved as 'reconstruction_comparison.png'")
    
    # Perform detailed evaluation
    print("\nPerforming detailed evaluation on validation data...")
    metrics = evaluate_model_detailed(model, val_loader)
    
    # Save metrics to file
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("Detailed Evaluation Results:\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")

    # Latent space visualization       
    print("\nVisualizing latent space...")
    features, labels, positions = visualize_latent_space(model, val_loader)

    # Add t-SNE visualization with sampling here
    print("\nGenerating t-SNE visualization with sampling...")
    tsne_result, sampled_indices = visualize_latent_space_tsne(
        features=features,
        labels=labels, 
        positions=positions,
        sample_size=500,  # Adjust based on your computational resources
        perplexity=30
    )
    
    print("\nAll visualizations and metrics have been saved.")

    return model, history

if __name__ == "__main__":
    model, history = main()

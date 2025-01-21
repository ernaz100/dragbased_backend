import torch
import torch.nn as nn
import os
from tqdm import tqdm

class PoseNetwork(nn.Module):
    def __init__(self, input_joints=24, joint_dims=3, hidden_size=1024, pose_params=72):
        """
        Neural network to predict SMPL pose parameters from 3D joint positions.
        
        Args:
            input_joints (int): Number of input joints (default: 24 for SMPL)
            joint_dims (int): Dimensions per joint (default: 3 for x,y,z)
            hidden_size (int): Size of hidden layers
            pose_params (int): Number of SMPL pose parameters to predict (default: 72)
        """
        super(PoseNetwork, self).__init__()
        
        self.network = nn.Sequential(
            # Flatten input joints
            nn.Flatten(),
            
            # First dense block
            nn.Linear(input_joints * joint_dims, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second dense block
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third dense block
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(hidden_size // 2, pose_params),
            # Note: No activation on final layer as pose parameters can be negative
        )

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_joints, 3)
            
        Returns:
            torch.Tensor: Predicted SMPL pose parameters of shape (batch_size, 72)
        """
        # Forward pass through network
        pose_params = self.network(x)
        return pose_params


def train_model(model, train_dataloader, val_dataloader, num_epochs=100, learning_rate=1e-4, checkpoint_dir='checkpoints'):
    """
    Training loop for the pose network with validation and model checkpointing.
    
    Args:
        model (PoseNetwork): The model to train
        train_dataloader (DataLoader): DataLoader containing the training data
        val_dataloader (DataLoader): DataLoader containing the validation data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimization
        checkpoint_dir (str): Directory to save model checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        for batch_idx, (joints, poses) in enumerate(train_pbar):
            joints = joints.to(device)
            poses = poses.to(device)
            
            # Forward pass
            predicted_poses = model(joints)
            
            # Compute loss
            loss = criterion(predicted_poses, poses)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        val_pbar = tqdm(val_dataloader, desc='Validation')
        with torch.no_grad():
            for joints, poses in val_pbar:
                joints = joints.to(device)
                poses = poses.to(device)
                
                predicted_poses = model(joints)
                val_loss = criterion(predicted_poses, poses)
                total_val_loss += val_loss.item()
                val_pbar.set_postfix({'val_loss': f'{val_loss.item():.4f}'})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print('Best Validation Loss: {:.4f}'.format(best_val_loss))
        print('-' * 50)

def evaluate_model(model, test_dataloader, device):
    """
    Evaluate model on test set.
    
    Args:
        model (PoseNetwork): The trained model
        test_dataloader (DataLoader): DataLoader containing the test data
        device (torch.device): Device to run evaluation on
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    test_pbar = tqdm(test_dataloader, desc='Testing')
    with torch.no_grad():
        for joints, poses in test_pbar:
            joints = joints.to(device)
            poses = poses.to(device)
            
            predicted_poses = model(joints)
            loss = criterion(predicted_poses, poses)
            total_loss += loss.item()
            test_pbar.set_postfix({'test_loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {avg_loss:.4f}')
    return avg_loss

if __name__ == "__main__":
    # Example usage
    from dataloader import AMASSDataset, DataLoader
    BATCH_SIZE = 32
    
    # Initialize datasets and dataloaders
    data_dirs = [
        "/Users/ericnazarenus/Desktop/dragbased/data/03099",
        "/Users/ericnazarenus/Desktop/dragbased/data/03100",
        "/Users/ericnazarenus/Desktop/dragbased/data/03101"
    ]
    
    # Combine all data directories for a larger dataset
    train_dataset = AMASSDataset(data_dirs[0])  # Start with first directory
    for dir in data_dirs[1:]:
        train_dataset.extend_dataset(dir) 
    
    # Calculate split sizes (70% train, 15% val, 15% test)
    total_size = len(train_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize and train model
    model = PoseNetwork()
    train_model(model, train_dataloader, val_dataloader, num_epochs=10)

    # Load best model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # Test the best model
    test_loss = evaluate_model(model, test_dataloader, device)
    
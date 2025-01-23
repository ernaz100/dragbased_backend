import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataloader import AMASSDataset
from sequence_dataloader import MotionSequenceDataset
from pose_estimator import PoseEstimator
from mpl_toolkits.mplot3d import Axes3D

class MotionSequenceNetwork(nn.Module):
    def __init__(self, input_joints=24, joint_dims=3, hidden_size=1024, pose_params=72):
        """
        Neural network to predict motion sequences given input sequence and target end position.
        
        Args:
            input_joints (int): Number of input joints
            joint_dims (int): Dimensions per joint
            hidden_size (int): Size of hidden layers
            pose_params (int): Number of SMPL pose parameters to predict
        """
        super(MotionSequenceNetwork, self).__init__()
        
        # Encoder for processing input sequence
        self.sequence_encoder = nn.GRU(
            input_size=input_joints * joint_dims,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Encoder for target end position
        self.target_encoder = nn.Sequential(
            nn.Linear(input_joints * joint_dims, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoder for generating pose sequences
        self.pose_decoder = nn.GRU(
            input_size=hidden_size * 2,  # Combined sequence and target encodings
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Final layers to predict pose parameters
        self.pose_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, pose_params)
        )

    def forward(self, input_sequence, target_position, output_sequence_length):
        """
        Forward pass of the network.
        
        Args:
            input_sequence (torch.Tensor): Input motion sequence [batch_size, seq_len, num_joints * 3]
            target_position (torch.Tensor): Target end position [batch_size, num_joints * 3]
            output_sequence_length (int): Length of sequence to generate
            
        Returns:
            torch.Tensor: Predicted pose parameters [batch_size, output_seq_len, 72]
        """
        batch_size = input_sequence.size(0)
        
        # Flatten joint dimensions for sequence processing
        flat_sequence = input_sequence.view(batch_size, -1, 72)
        
        # Encode input sequence
        _, sequence_encoding = self.sequence_encoder(flat_sequence)
        sequence_encoding = sequence_encoding[-1]  # Take last layer's hidden state
        
        # Encode target position
        flat_target = target_position.view(batch_size, -1)
        target_encoding = self.target_encoder(flat_target)
        
        # Combine encodings
        combined_encoding = torch.cat([sequence_encoding, target_encoding], dim=-1)
        
        # Prepare decoder input (repeat combined encoding for each output timestep)
        decoder_input = combined_encoding.unsqueeze(1).repeat(1, output_sequence_length, 1)
        
        # Generate sequence
        decoder_output, _ = self.pose_decoder(decoder_input)
        
        # Predict pose parameters for each timestep
        pose_sequence = self.pose_predictor(decoder_output)
        
        return pose_sequence

def poses_to_joints(poses, pose_estimator):
    """
    Convert SMPL pose parameters to joint positions using forward kinematics
    
    Args:
        poses: [seq_len, 72] tensor of SMPL pose parameters
        pose_estimator: PoseEstimator instance
        
    Returns:
        [seq_len, 24, 3] tensor of joint positions
    """
    seq_len = poses.size(0)
    joints = []
    
    for i in range(seq_len):
        pose = poses[i].detach().cpu().numpy()
        joints_3d = pose_estimator.forward_kinematics(pose)
        joints.append(joints_3d)
    
    return torch.tensor(np.stack(joints))

def train_sequence_model(model, train_dataloader, val_dataloader, num_epochs=100, 
                        learning_rate=1e-4, checkpoint_dir='sequence_checkpoints', 
                        checkpoint_name='best_sequence_model.pth'):
    """
    Training loop for the sequence network with visualization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        for batch in train_pbar:
            input_sequence = batch['input_sequence'].to(device)
            target_position = batch['target_end_position'].to(device)
            ground_truth_poses = batch['ground_truth_poses'].to(device)
            # Forward pass
            predicted_poses = model(
                input_sequence.view(input_sequence.size(0), -1, 72),
                target_position,
                ground_truth_poses.size(1)
            )
            
            # Compute loss
            loss = criterion(predicted_poses, ground_truth_poses)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_sequence = batch['input_sequence'].to(device)
                target_position = batch['target_end_position'].to(device)
                ground_truth_poses = batch['ground_truth_poses'].to(device)
                
                predicted_poses = model(
                    input_sequence.view(input_sequence.size(0), -1, 72),
                    target_position,
                    ground_truth_poses.size(1)
                )
                
                loss = criterion(predicted_poses, ground_truth_poses)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_dir, checkpoint_name))
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'sequence_loss_plot.png'))
    plt.close()
    return train_losses, val_losses 

if __name__ == "__main__":
    data_dirs = [
        "/Users/ericnazarenus/Desktop/dragbased/data/03099",
        "/Users/ericnazarenus/Desktop/dragbased/data/03100",
        "/Users/ericnazarenus/Desktop/dragbased/data/03101"
    ]
    INPUT_FRAMES_LEN = 30 # 4 seconds at 30 fps
    OUTPUT_FRAMES_LEN = 30 # 1 second prediction to go from last input frame to defined joint positions
    EPOCHS = 10
    # Create base dataset
    amass_dataset = AMASSDataset(data_dirs[0])
    for dir in data_dirs[1:]:
        amass_dataset.extend_dataset(dir)
    
    # Create sequence dataset
    sequence_dataset = MotionSequenceDataset(
        amass_dataset,
        input_sequence_length=INPUT_FRAMES_LEN,  
        output_sequence_length=OUTPUT_FRAMES_LEN  
    )
    
    # Split dataset
    total_size = len(sequence_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        sequence_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    BATCH_SIZE = 32
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize and train model
    model = MotionSequenceNetwork()
    train_sequence_model(model, train_dataloader, val_dataloader, 
                        num_epochs=EPOCHS, learning_rate=1e-4)
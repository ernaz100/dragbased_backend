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
import imageio
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

def visualize_sequence(true_seq, pred_seq, frame_idx, save_path):
    """
    Visualize true and predicted sequences side by side, showing a single frame
    
    Args:
        true_seq: [seq_len, num_joints, 3] tensor of true joint positions
        pred_seq: [seq_len, num_joints, 3] tensor of predicted joint positions
        frame_idx: Frame number for filename
        save_path: Directory to save frames
    """
    # Create a figure for each frame in the sequence
    for t in range(true_seq.shape[0]):
        fig = plt.figure(figsize=(12, 6))
        
        # True sequence plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=20, azim=45)
        ax1.set_title(f'Ground Truth - Frame {t+1}')
        
        # Predicted sequence plot
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=20, azim=45)
        ax2.set_title(f'Predicted - Frame {t+1}')
        
        # Set consistent axis limits and labels
        all_points = torch.cat([true_seq, pred_seq], dim=0)
        min_val = all_points.min().item()
        max_val = all_points.max().item()
        
        for ax in [ax1, ax2]:
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_zlim(min_val, max_val)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        # Get current frame joints
        true_joints = true_seq[t].detach().cpu().numpy()
        pred_joints = pred_seq[t].detach().cpu().numpy()
        
        # Define SMPL kinematic tree connections
        connections = [
            (0, 1), (1, 4), (4, 7), (7, 10),  # Left leg
            (0, 2), (2, 5), (5, 8), (8, 11),  # Right leg
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # Spine and head
            (9, 13), (13, 16), (16, 18), (18, 20),  # Left arm
            (9, 14), (14, 17), (17, 19), (19, 21)   # Right arm
        ]
        
        # Plot ground truth frame
        ax1.scatter(true_joints[:, 0], true_joints[:, 1], true_joints[:, 2], 
                   c='b', marker='o', s=50)
        for connection in connections:
            start, end = connection
            ax1.plot([true_joints[start, 0], true_joints[end, 0]],
                    [true_joints[start, 1], true_joints[end, 1]],
                    [true_joints[start, 2], true_joints[end, 2]], 
                    'b-')
        
        # Plot predicted frame
        ax2.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], 
                   c='r', marker='o', s=50)
        for connection in connections:
            start, end = connection
            ax2.plot([pred_joints[start, 0], pred_joints[end, 0]],
                    [pred_joints[start, 1], pred_joints[end, 1]],
                    [pred_joints[start, 2], pred_joints[end, 2]], 
                    'r-')
        
        plt.savefig(os.path.join(save_path, f'frame_{frame_idx:04d}_seq_{t:04d}.png'))
        plt.close()

def train_sequence_model(model, train_dataloader, val_dataloader, num_epochs=100, 
                        learning_rate=1e-4, checkpoint_dir='checkpoints', 
                        checkpoint_name='best_sequence_model.pth'):
    """
    Training loop for the sequence network with visualization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(model_path="../models/smpl")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create visualization directories
    vis_dir = os.path.join(checkpoint_dir, 'visualizations')
    frames_dir = os.path.join(vis_dir, 'frames')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Get a fixed batch for visualization
    vis_batch = next(iter(val_dataloader))
    vis_input = vis_batch['input_sequence'].to(device)
    vis_target = vis_batch['target_end_position'].to(device)
    vis_ground_truth = vis_batch['ground_truth_poses'].to(device)
    
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
            
            # Visualization step
            pred_poses = model(
                vis_input.view(vis_input.size(0), -1, 72),
                vis_target,
                vis_ground_truth.size(1)
            )
            
            # Convert poses to joint positions using forward kinematics
            true_joints = poses_to_joints(vis_ground_truth[0], pose_estimator)
            pred_joints = poses_to_joints(pred_poses[0], pose_estimator)
            
            # Create visualization frame
            visualize_sequence(true_joints, pred_joints, epoch, frames_dir)
        
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
    
    # Create final visualization GIF
    create_gif(frames_dir, os.path.join(vis_dir, 'sequence_evolution.gif'))
    
    return train_losses, val_losses 

def create_gif(frame_dir, output_path, duration=0.5):
    """
    Create a looping GIF from a directory of frames
    
    Args:
        frame_dir: Directory containing the frame images
        output_path: Path where the GIF will be saved
        duration: Duration for each frame in seconds
    """
    frames = []
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frames.append(imageio.imread(frame_path))
    
    # Save with loop=0 to make it loop indefinitely
    imageio.mimsave(output_path, frames, duration=duration, loop=0)

if __name__ == "__main__":
    data_dirs = [
        "/Users/ericnazarenus/Desktop/dragbased/data/03099",
        "/Users/ericnazarenus/Desktop/dragbased/data/03100",
        "/Users/ericnazarenus/Desktop/dragbased/data/03101"
    ]
    INPUT_FRAMES_LEN = 120 # 4 seconds at 30 fps
    OUTPUT_FRAMES_LEN = 30 # 1 second prediction to go from last input frame to defined joint positions
    EPOCHS = 25
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
    BATCH_SIZE = 64
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize and train model
    model = MotionSequenceNetwork()
    train_sequence_model(model, train_dataloader, val_dataloader, 
                        num_epochs=EPOCHS, learning_rate=1e-4)
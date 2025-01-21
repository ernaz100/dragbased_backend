import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataloader import AMASSDataset

class MotionSequenceDataset(Dataset):
    def __init__(self, amass_dataset, input_sequence_length=30, output_sequence_length=30):
        """
        Dataset for sequence-to-sequence motion prediction.
        
        Args:
            amass_dataset (AMASSDataset): Base AMASS dataset
            input_sequence_length (int): Number of input frames
            output_sequence_length (int): Number of output frames to predict
        """
        self.amass_dataset = amass_dataset
        self.input_seq_len = input_sequence_length
        self.output_seq_len = output_sequence_length
        
        # Create valid sequence indices
        self.sequence_indices = []
        current_idx = 0
        
        for poses in self.amass_dataset.all_poses:
            # Each sequence needs: input_sequence + output_sequence frames
            total_required = input_sequence_length + output_sequence_length
            if len(poses) >= total_required:
                # Create sequences with overlap for more training data
                for i in range(len(poses) - total_required + 1):
                    self.sequence_indices.append((current_idx, i))
            current_idx += len(poses)

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        file_idx, start_idx = self.sequence_indices[idx]
        
        # Get input sequence
        input_joints = []
        for i in range(self.input_seq_len):
            joints, _ = self.amass_dataset[start_idx + i]
            input_joints.append(joints)
        
        # Get target end position (last frame of output sequence)
        target_joints, _ = self.amass_dataset[start_idx + self.input_seq_len + self.output_seq_len - 1]
        
        # Get ground truth output sequence
        output_joints = []
        output_poses = []
        for i in range(self.output_seq_len):
            joints, poses = self.amass_dataset[start_idx + self.input_seq_len + i]
            output_joints.append(joints)
            output_poses.append(poses)
        
        # Stack sequences into tensors
        input_sequence = torch.stack(input_joints)  # [input_seq_len, num_joints, 3]
        output_sequence = torch.stack(output_joints)  # [output_seq_len, num_joints, 3]
        output_poses = torch.stack(output_poses)  # [output_seq_len, 72]
        
        return {
            'input_sequence': input_sequence,
            'target_end_position': target_joints,
            'ground_truth_sequence': output_sequence,
            'ground_truth_poses': output_poses
        } 
    

if __name__ == "__main__":
    data_dir = "/Users/ericnazarenus/Desktop/dragbased/data/03099"

    # Initialize the base AMASS dataset
    amass_dataset = AMASSDataset(data_dir)
    
    # Create the sequence dataset
    sequence_dataset = MotionSequenceDataset(
        amass_dataset,
        input_sequence_length=30,  # 1 second at 30 fps
        output_sequence_length=30  # 1 second prediction
    )
    
    # Create dataloader for the sequence dataset
    dataloader = DataLoader(sequence_dataset, batch_size=32, shuffle=True)

    # Get first batch
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Input Sequence Shape: {batch['input_sequence'].shape}")
        print(f"Target End Position Shape: {batch['target_end_position'].shape}")
        print(f"Ground Truth Sequence Shape: {batch['ground_truth_sequence'].shape}")
        print(f"Ground Truth Poses Shape: {batch['ground_truth_poses'].shape}")
        
        break  # Only process first batch
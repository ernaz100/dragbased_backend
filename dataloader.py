import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from smplx import SMPL
from pose_estimator import PoseEstimator

class AMASSDataset(Dataset):
    def __init__(self, data_dir, normalize_joints=True, model_path="../models/smpl", sequence_length=1):
        """
        AMASS Dataset for loading 3D joint positions and SMPL pose parameters.

        Args:
            data_dir (str): Path to the directory containing AMASS `.npz` files.
            normalize_joints (bool): Whether to normalize joints relative to the pelvis.
            model_path (str): Path to the SMPL model files
            sequence_length (int): Fixed length for all sequences. Longer sequences will be cut,
                                 shorter ones will be padded.
        """
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.normalize_joints = normalize_joints
        self.sequence_length = sequence_length
        # Initialize SMPL model from smplx
        self.mesh_model = SMPL(model_path, gender='female')
        self.smpl_root_joint_idx = 0  # Pelvis is typically index 0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)

        # Extract SMPL parameters
        poses = data["poses"]  # (N_frames, 156)
        betas = data["betas"][:10]  # Get first 10 shape parameters
        
        # Process frames
        joints_list = []
        poses_list = []
        
        # Limit number of frames to sequence_length
        n_frames = min(len(poses), self.sequence_length)
        print(len(poses))
        for frame_idx in range(n_frames):
            # Get pose parameters for current frame (first 72 values for pose)
            pose = poses[frame_idx:frame_idx+1, :72]
            beta = betas[None, :]  # Add batch dimension
            
            # Convert to torch tensors
            smpl_pose = torch.FloatTensor(pose)
            smpl_shape = torch.FloatTensor(beta)
            
            # Get joint coordinates using smplx
            output = self.mesh_model(
                betas=smpl_shape,
                body_pose=smpl_pose[:, 3:],  # body pose
                global_orient=smpl_pose[:, :3]  # global orientation
            )
            
            # Extract joints from output
            joints = output.joints.detach().numpy().squeeze()[:24]
            
            # Normalize joints if requested
            if self.normalize_joints:
                root_joint = joints[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
                joints = joints - root_joint
            
            joints_list.append(joints)
            poses_list.append(pose.reshape(-1))

        # Stack frames
        joints = np.stack(joints_list)
        poses = np.stack(poses_list)

        # Pad sequences if they're shorter than sequence_length
        if n_frames < self.sequence_length:
            # Pad with zeros
            joints_pad = np.zeros((self.sequence_length - n_frames, joints.shape[1], joints.shape[2]))
            poses_pad = np.zeros((self.sequence_length - n_frames, poses.shape[1]))
            
            joints = np.concatenate([joints, joints_pad], axis=0)
            poses = np.concatenate([poses, poses_pad], axis=0)

        # Convert to tensors
        joints = torch.tensor(joints, dtype=torch.float32)
        poses = torch.tensor(poses, dtype=torch.float32)

        return joints, poses

data_dir = "/Users/ericnazarenus/Desktop/dragbased/data/03099"

# Initialize the dataset and dataloader
dataset = AMASSDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=13, shuffle=True)  # Set batch_size to 1 for visualization
pose_estimator = PoseEstimator()

if __name__ == "__main__":
    file_path = "/Users/ericnazarenus/Desktop/dragbased/data/03101/ROM2_poses.npz"
    data = np.load(file_path)

    print("Keys in the .npz file:", list(data.keys()))

    # Get first batch
    for batch_idx, (joints, poses) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Joints Shape: {joints.shape}")
        print(f"Poses Shape: {poses.shape}")

        # Convert first frame of joints to numpy for visualization
        joints_np = joints[0, 0].numpy()  # Get first frame of first batch
        poses_np = poses[0, 0].numpy()    # Get first frame of first batch
        print(joints_np)
        # Visualize joints
        pose_estimator.visualize_joints(joints_np, title="amass_joints.png")
        
        # Visualize SMPL mesh with poses
        pose_estimator.visualize_pose(poses_np, title="amass_pose.png")
        
        # Optionally, export as GLB
        pose_estimator.export_pose_glb(poses_np, "amass_pose.glb")
        
        break  # Only process first batch

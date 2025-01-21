import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from smplx import SMPL
from pose_estimator import PoseEstimator

class AMASSDataset(Dataset):
    def __init__(self, data_dir, normalize_joints=True, model_path="../models/smpl"):
        """
        AMASS Dataset for loading 3D joint positions and SMPL pose parameters.

        Args:
            data_dir (str): Path to the directory containing AMASS `.npz` files.
            normalize_joints (bool): Whether to normalize joints relative to the pelvis.
            model_path (str): Path to the SMPL model files
        """
        self.data_dir = data_dir
        self.normalize_joints = normalize_joints
        
        # Load all poses and their file indices upfront
        self.all_poses = []
        self.all_betas = []
        
        for file_idx, file_path in enumerate(os.listdir(data_dir)):
            if file_path.endswith('.npz'):
                data = np.load(os.path.join(data_dir, file_path))
                poses = data["poses"]  # (N_frames, 156)
                betas = data["betas"][:10]  # Get first 10 shape parameters
                
                # Store poses and corresponding betas
                self.all_poses.append(poses)
                self.all_betas.append(betas)
        
        # Calculate total number of frames across all files
        self.total_frames = sum(len(poses) for poses in self.all_poses)
        
        # Initialize SMPL model from smplx
        self.mesh_model = SMPL(model_path, gender='female')
        self.smpl_root_joint_idx = 0  # Pelvis is typically index 0

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # Find which file and frame this index corresponds to
        for file_idx, poses in enumerate(self.all_poses):
            if idx < len(poses):
                break
            idx -= len(poses)
        
        # Get pose and beta for this frame
        pose = self.all_poses[file_idx][idx, :72]  # Get first 72 values for pose
        beta = self.all_betas[file_idx][None, :]  # Add batch dimension
        
        # Convert to torch tensors
        smpl_pose = torch.FloatTensor(pose[None, :])
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
        
        # Convert to tensors
        joints = torch.tensor(joints, dtype=torch.float32)
        poses = torch.tensor(pose, dtype=torch.float32)

        return joints, poses

data_dir = "/Users/ericnazarenus/Desktop/dragbased/data/03099"

# Initialize the dataset and dataloader
dataset = AMASSDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Adjust batch_size as needed
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
        joints_np = joints[0].numpy()  # Get first frame of first batch
        poses_np = poses[0].numpy()    # Get first frame of first batch
        print(joints_np)
        # Visualize joints
        pose_estimator.visualize_joints(joints_np, title="amass_joints.png")
        
        # Visualize SMPL mesh with poses
        pose_estimator.visualize_pose(poses_np, title="amass_pose.png")
        
        # Optionally, export as GLB
        pose_estimator.export_pose_glb(poses_np, "amass_pose.glb")
        
        break  # Only process first batch

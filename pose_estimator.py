import numpy as np
import torch
from smplx import SMPL
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import os

class PoseEstimator:
    def __init__(self, model_path="/Users/ericnazarenus/Desktop/dragbased/models/smpl"):
        self.smpl_model = SMPL(model_path, gender='male')
        # Add SMPL joint mapping
        self.joint_mapping = {
            0: 0,   # root
            1: 0,   # pelvis (same as root in SMPL)
            2: 1,   # left hip
            3: 4,   # left knee
            4: 7,   # left ankle
            5: 10,  # left foot
            6: 2,   # right hip
            7: 5,   # right knee
            8: 8,   # right ankle
            9: 11,  # right foot
            10: 3,  # spine1
            11: 6,  # spine2
            12: 9,  # spine3
            13: 12, # neck
            14: 15, # head
            15: 13, # left collar
            16: 16, # left shoulder
            17: 18, # left elbow
            18: 20, # left wrist
            19: 22, # left hand
            20: 14, # right collar
            21: 17, # right shoulder
            22: 19, # right elbow
            23: 21, # right wrist
            24: 23, # right hand
        }

    def forward_kinematics(self, pose_params):
        """ Compute joint positions from given pose parameters """
        # Separate global orientation (root) and body pose
        global_orient = pose_params[:3].reshape(1, 3)
        body_pose = pose_params[3:].reshape(1, 69)
        
        # Convert to torch tensors
        global_orient = torch.tensor(global_orient, dtype=torch.float32)
        body_pose = torch.tensor(body_pose, dtype=torch.float32)
        
        output = self.smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=torch.zeros(1, 10),
            return_verts=False
        )
        return output.joints[0].detach().cpu().numpy()

    def loss_function(self, pose_params, target_joints):
        """ Compute the loss between current and target joint positions """
        predicted_joints = self.forward_kinematics(pose_params)
        return np.linalg.norm(predicted_joints[:24] - target_joints)

    def estimate_pose(self, target_joints):
        """ Optimize pose to match given joint positions """
        # Remap joints to SMPL ordering
        remapped_joints = self.remap_joints(target_joints)
        
        # Initialize with 72 parameters (3 for global orientation + 23 joints * 3 rotation params)
        initial_pose = np.zeros(72)
        
        # Add bounds to keep rotations reasonable
        bounds = [(-np.pi, np.pi)] * 72
        
        result = minimize(
            self.loss_function, 
            initial_pose, 
            args=(remapped_joints,), 
            method='L-BFGS-B',
            bounds=bounds
        )
        return result.x

    def visualize_pose(self, pose_params=None, title="pose_visualization.png"):
        """Visualize the SMPL model pose using Matplotlib and save as image"""
        if pose_params is None:
            pose_params = np.zeros(72)  # Default T-pose
            
        # Get joints and vertices
        global_orient = torch.tensor(pose_params[:3].reshape(1, 3), dtype=torch.float32)
        body_pose = torch.tensor(pose_params[3:].reshape(1, 69), dtype=torch.float32)
        
        output = self.smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=torch.zeros(1, 10),
            return_verts=True
        )
        
        vertices = output.vertices[0].detach().cpu().numpy()
        joints = output.joints[0].detach().cpu().numpy()
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 5))
        
        # Front view - swap coordinates to match visualize_joints
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(vertices[:, 2], vertices[:, 0], vertices[:, 1], c='b', marker='.', s=1)
        ax1.scatter(joints[:, 2], joints[:, 0], joints[:, 1], c='r', marker='o')
        ax1.view_init(elev=0, azim=0)
        ax1.set_title('Front View')
        
        # Side view
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(vertices[:, 2], vertices[:, 0], vertices[:, 1], c='b', marker='.', s=1)
        ax2.scatter(joints[:, 2], joints[:, 0], joints[:, 1], c='r', marker='o')
        ax2.view_init(elev=0, azim=90)
        ax2.set_title('Side View')
        
        # Top view
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(vertices[:, 2], vertices[:, 0], vertices[:, 1], c='b', marker='.', s=1)
        ax3.scatter(joints[:, 2], joints[:, 0], joints[:, 1], c='r', marker='o')
        ax3.view_init(elev=90, azim=0)
        ax3.set_title('Top View')
        
        # Set consistent axes limits
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1,1,1])
        
        # Save the figure
        plt.tight_layout()
        save_path = title
        plt.savefig(save_path)
        plt.close()
        
        return save_path

    def visualize_joints(self, joints, title="input_joints.png"):
        """Visualize input joint positions"""
        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 5))
        
        # Front view - swap x and z coordinates for correct orientation
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(joints[:, 2], joints[:, 0], joints[:, 1], c='r', marker='o')
        ax1.view_init(elev=0, azim=0)
        ax1.set_title('Front View')
        
        # Side view
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(joints[:, 2], joints[:, 0], joints[:, 1], c='r', marker='o')
        ax2.view_init(elev=0, azim=90)
        ax2.set_title('Side View')
        
        # Top view
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(joints[:, 2], joints[:, 0], joints[:, 1], c='r', marker='o')
        ax3.view_init(elev=90, azim=0)
        ax3.set_title('Top View')
        
        # Set consistent axes limits and labels
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1,1,1])
        
        # Save the figure
        plt.tight_layout()
        save_path = title
        plt.savefig(save_path)
        plt.close()
        
        return save_path

    def export_pose_glb(self, pose_params=None, output_path="pose.glb"):
        """Export the SMPL model pose as a GLB file with joint positions"""
        if pose_params is None:
            pose_params = np.zeros(72)  # Default T-pose
            
        # Convert pose parameters to torch tensors with correct shape and type
        global_orient = torch.tensor(pose_params[:3].reshape(1, 3), dtype=torch.float32)
        body_pose = torch.tensor(pose_params[3:].reshape(1, 69), dtype=torch.float32)
        
        # Get model output with vertices and joints
        with torch.no_grad():
            output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=torch.zeros((1, 10), dtype=torch.float32),
                return_verts=True,
                return_full_pose=True
            )
        
        # Get vertices, faces, and joints
        vertices = output.vertices[0].detach().cpu().numpy()
        faces = self.smpl_model.faces
        joints = output.joints[0].detach().cpu().numpy()
        
        # Create mesh for the body
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create small spheres for joints
        joint_meshes = []
        for joint_pos in joints:
            joint_sphere = trimesh.primitives.Sphere(radius=0.02, center=joint_pos)
            joint_meshes.append(joint_sphere)
        
        # Combine body mesh with joint spheres
        combined_mesh = trimesh.util.concatenate([mesh] + joint_meshes)
        
        # Rotate mesh 90 degrees around X-axis to match standard orientation
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        combined_mesh.apply_transform(rotation)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        combined_mesh.export(output_path)
        return output_path

    def remap_joints(self, input_joints):
        """
        Remap joints from frontend ordering to SMPL ordering
        
        Args:
            input_joints: numpy array of shape (25, 3) in frontend order
        
        Returns:
            remapped_joints: numpy array of shape (24, 3) in SMPL order
        """
        remapped_joints = np.zeros((24, 3))
        for frontend_idx, smpl_idx in self.joint_mapping.items():
            if frontend_idx < len(input_joints):
                remapped_joints[smpl_idx] = input_joints[frontend_idx]
        return remapped_joints

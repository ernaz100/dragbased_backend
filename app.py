from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from pose_estimator import PoseEstimator
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from pose_network import PoseNetwork

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize pose estimator
pose_estimator = PoseEstimator()
joint_mapping = {
            0: 0,   # pelvis 
            1: 1,   # left hip 
            2: 4,   # left knee
            3: 7,   # left ankle
            4: 10,  # left foot
            5: 2,   # right hip
            6: 5,   # right knee
            7: 8,   # right ankle
            8: 11,  # right foot
            9: 3,  # spine1
            10: 6,  # spine2
            11: 9,  # spine3
            12: 12, # neck
            13: 15, # head
            14: 13, # left collar
            15: 16, # left shoulder
            16: 18, # left elbow
            17: 20, # left wrist
            18: 22, # left hand
            19: 14, # right collar
            20: 17, # right shoulder
            21: 19, # right elbow
            22: 21, # right wrist
            23: 23, # right hand
        }

# Initialize pose network
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
pose_network = PoseNetwork()
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True)
pose_network.load_state_dict(checkpoint['model_state_dict'])
pose_network.to(device)
pose_network.eval()

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/estimate_pose', methods=['POST'])
def handle_pose_estimation():
    try:
        data = request.get_json()
        logger.info("Received pose estimation request")
        
        if 'joint_positions' not in data:
            logger.error("No joint positions provided in request")
            return jsonify({'error': 'No joint positions provided'}), 400
            
        joint_positions = np.array(data['joint_positions'])
        selected_joint = data['selected_joint'] 
        
        logger.info(f"Received joint positions with shape: {joint_positions.shape}")
        logger.info(f"Selected joint: {selected_joint}")
        
        # Validate input shape
        if joint_positions.shape != (24, 3):
            logger.error(f"Invalid joint positions shape: {joint_positions.shape}")
            return jsonify({'error': 'Invalid joint positions format. Expected shape: (24, 3)'}), 400

        # Remap joints to SMPL order
        remapped_joints = remap_joints(joint_positions)
        
        # Prepare input for pose network
        joints_tensor = torch.tensor(remapped_joints, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        joints_tensor = joints_tensor.to(device)
        
        # Get pose parameters from network
        with torch.no_grad():
            pose_params = pose_network(joints_tensor)
            pose_params = pose_params.cpu().numpy().squeeze()  # Remove batch and sequence dims
        frontend_pose_params = remap_pose_params_back(pose_params)

        glb_path = pose_estimator.export_pose_glb(frontend_pose_params, "static/optimized_pose_net.glb")
        output_viz_path = pose_estimator.visualize_pose(pose_params=frontend_pose_params, title="optimized_pose_net.png" ,selected_joint=selected_joint)
        result = {
            'pose_params': frontend_pose_params.tolist(),
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during pose estimation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def remap_joints(input_joints):
        """
        Remap joints from frontend ordering to SMPL ordering
        
        Args:
            input_joints: numpy array of shape (24, 3) in frontend order
        
        Returns:
            remapped_joints: numpy array of shape (24, 3) in SMPL order
        """
        if input_joints.shape != (24, 3):
            raise ValueError(f"Expected input_joints shape (24, 3), got {input_joints.shape}")

        remapped_joints = np.zeros((24, 3))
        for frontend_idx, smpl_idx in joint_mapping.items():
            remapped_joints[smpl_idx] = input_joints[frontend_idx]
        return remapped_joints

def remap_pose_params_back(smpl_pose_params):
    """
    Remap pose parameters from SMPL ordering back to frontend ordering
    
    Args:
        smpl_pose_params: numpy array of shape (72,) in SMPL order
        (first 3 values are global orientation, then 23 joints * 3 rotation params)
    
    Returns:
        frontend_pose_params: numpy array of shape (72,) in frontend order
    """
    if smpl_pose_params.shape != (72,):
        raise ValueError(f"Expected smpl_pose_params shape (72,), got {smpl_pose_params.shape}")

    # Create output array
    frontend_pose_params = np.zeros(72)
    
    # Remap the joint rotations (remaining 69 values, 3 per joint)
    for frontend_idx, smpl_idx in joint_mapping.items():
        src_idx = smpl_idx * 3 
        dst_idx = frontend_idx * 3 
        frontend_pose_params[dst_idx:dst_idx] = smpl_pose_params[src_idx:src_idx]
    
    return frontend_pose_params


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 

            # # Visualize input joints with selected joint highlighted
        # input_viz_path = pose_estimator.visualize_joints(joint_positions, selected_joint=selected_joint, title="input_joints.png")
        
        # # Estimate pose with selected joint constraint
        # joint_pos, optimized_pose = pose_estimator.estimate_pose(joint_positions, selected_joint=selected_joint)
        # logger.info("Pose estimation completed successfully")
        # # Visualize and save the optimized pose
        # output_viz_path = pose_estimator.visualize_pose(pose_params=optimized_pose, title="optimized_pose.png" ,selected_joint=selected_joint)
        # # Export the pose as GLB
        # glb_path = pose_estimator.export_pose_glb(optimized_pose, "static/optimized_pose.glb")
        
        # result = {
        #     'pose_params': optimized_pose.tolist(),
        #     'joint_pos': joint_pos.tolist(),
        #     'glb_url': glb_path,  # URL path to the GLB file
        #     'status': 'success'
        # }

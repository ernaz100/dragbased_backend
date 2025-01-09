from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from pose_estimator import PoseEstimator
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize pose estimator
pose_estimator = PoseEstimator()

# Add this route to serve static files
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
        logger.info(f"Received joint positions with shape: {joint_positions.shape}")
        
        # Validate input shape
        if joint_positions.shape != (24, 3):
            logger.error(f"Invalid joint positions shape: {joint_positions.shape}")
            return jsonify({'error': 'Invalid joint positions format. Expected shape: (24, 3)'}), 400
            
        # Visualize input joints
        input_viz_path = pose_estimator.visualize_joints(joint_positions, title="input_joints.png")
        
        # Estimate pose
        joint_pos ,optimized_pose = pose_estimator.estimate_pose(joint_positions)
        logger.info("Pose estimation completed successfully")
        # Visualize and save the optimized pose
        output_viz_path = pose_estimator.visualize_pose(optimized_pose, title="optimized_pose.png")
        # Export the pose as GLB
        glb_path = pose_estimator.export_pose_glb(optimized_pose, "static/optimized_pose.glb")
        
        result = {
            'pose_params': optimized_pose.tolist(),
            'glb_url': glb_path,  # URL path to the GLB file
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during pose estimation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 
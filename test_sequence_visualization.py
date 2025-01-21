import torch
from sequence_network import MotionSequenceNetwork, train_sequence_model
from pose_estimator import PoseEstimator
from sequence_dataloader import MotionSequenceDataset
from dataloader import AMASSDataset
from torch.utils.data import DataLoader, Subset

def test_visualization():
    print("Starting visualization test...")
    
    # 1. Create a small test dataset
    print("Creating test dataset...")
    try:
        data_dir = "/Users/ericnazarenus/Desktop/dragbased/data/03099"  # Use your actual path
        amass_dataset = AMASSDataset(data_dir)
        sequence_dataset = MotionSequenceDataset(
            amass_dataset,
            input_sequence_length=100,
            output_sequence_length=10
        )
        
        # Take only first 100 samples for testing
        subset_indices = list(range(100))  # Just use 100 samples
        sequence_dataset = Subset(sequence_dataset, subset_indices)
        
        # Create small train/val split
        train_size = int(0.8 * len(sequence_dataset))
        val_size = len(sequence_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            sequence_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Dataset created successfully. Size: {len(sequence_dataset)}")
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return
    
    # 2. Create dataloaders
    print("Creating dataloaders...")
    try:
        batch_size = 2  # Small batch size for testing
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print("Dataloaders created successfully")
    except Exception as e:
        print(f"Error creating dataloaders: {str(e)}")
        return
    
    # 3. Initialize model
    print("Initializing model...")
    try:
        model = MotionSequenceNetwork()
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return
    
    # 4. Test pose estimator
    print("Testing pose estimator...")
    try:
        pose_estimator = PoseEstimator(model_path="../models/smpl")
        # Test forward kinematics with zero pose
        test_pose = torch.zeros(72)
        joints = pose_estimator.forward_kinematics(test_pose.numpy())
        print(f"Pose estimator working. Output joint shape: {joints.shape}")
    except Exception as e:
        print(f"Error testing pose estimator: {str(e)}")
        return
    
    # 5. Test training loop with minimal epochs
    print("Testing training loop...")
    try:
        train_sequence_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=2,  # Just 2 epochs for testing
            learning_rate=1e-4,
            checkpoint_dir='test_checkpoints',
            checkpoint_name='test_model.pth'
        )
        print("Training loop completed successfully")
    except Exception as e:
        print(f"Error in training loop: {str(e)}")
        return
    
    print("\nAll tests completed successfully!")
    print("Check the test_checkpoints/visualizations directory for the output GIF")

if __name__ == "__main__":
    test_visualization() 
#!/usr/bin/env python3
"""Test script for OccupancyVoxelMap integration in VLFM.

This script demonstrates how to use the newly integrated OccupancyVoxelMap
from RayFronts within the VLFM framework.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import vlfm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from vlfm.mapping.occ_voxel_map import OccupancyVoxelMap


def create_synthetic_depth_image(height=240, width=320, device='cuda'):
    """Create a synthetic depth image for testing."""
    # Create a simple depth image with some obstacles
    depth = torch.ones((height, width), device=device) * 5.0  # 5 meters default
    
    # Add some closer obstacles
    depth[60:180, 80:120] = 2.0   # Vertical wall
    depth[120:140, 150:250] = 1.5  # Horizontal obstacle
    depth[40:80, 200:280] = 3.0    # Another obstacle
    
    # Add some noise
    noise = torch.randn_like(depth) * 0.1
    depth = depth + noise
    depth = torch.clamp(depth, 0.1, 10.0)
    
    return depth


def create_camera_intrinsics(width=320, height=240):
    """Create camera intrinsics matrix."""
    fx = fy = 200.0  # Focal length
    cx = width / 2.0
    cy = height / 2.0
    
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return intrinsics


def create_camera_pose(x=0, y=0, z=1.5, yaw=0):
    """Create a 4x4 camera pose matrix."""
    # Simple pose: camera at (x, y, z) looking forward
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    pose = torch.tensor([
        [cos_yaw, 0, sin_yaw, x],
        [0, 1, 0, y],
        [-sin_yaw, 0, cos_yaw, z],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    
    return pose


def test_occ_voxel_map():
    """Test the OccupancyVoxelMap functionality."""
    print("Testing OccupancyVoxelMap integration...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create the occupancy voxel map
    voxel_map = OccupancyVoxelMap(
        vox_size=0.1,  # 10cm voxels
        max_pts_per_frame=5000,
        max_empty_pts_per_frame=2000,
        device=device,
        clip_bbox=((-10, -10, -2), (10, 10, 5)),  # Limit mapping region
        size=1000,
        pixels_per_meter=20
    )
    
    print(f"Created voxel map with voxel size: {voxel_map.vox_size}m")
    print(f"Initial voxel count: {voxel_map.get_voxel_count()}")
    
    # Create camera parameters
    intrinsics = create_camera_intrinsics().to(device)
    print(f"Camera intrinsics:\n{intrinsics}")
    
    # Test with multiple camera poses
    poses = [
        create_camera_pose(0, 0, 1.5, 0),      # Looking forward
        create_camera_pose(1, 0, 1.5, 0.5),   # Step right and turn
        create_camera_pose(0, 1, 1.5, -0.3),  # Step forward and turn left
    ]
    
    for i, pose in enumerate(poses):
        print(f"\n--- Processing frame {i+1} ---")
        
        # Create synthetic depth image
        depth = create_synthetic_depth_image(device=device)
        print(f"Depth image shape: {depth.shape}")
        print(f"Depth range: {depth.min().item():.2f} - {depth.max().item():.2f}m")
        
        # Update the map
        voxel_map.update_map(
            depth=depth,
            tf_camera_to_episodic=pose.numpy(),
            intrinsics=intrinsics.cpu().numpy(),
            min_depth=0.1,
            max_depth=10.0
        )
        
        print(f"Voxel count after frame {i+1}: {voxel_map.get_voxel_count()}")
        
        # Get occupied voxels
        occupied_xyz, occupied_occ = voxel_map.get_occupied_voxels(threshold=0.0)
        print(f"Occupied voxels (threshold=0.0): {occupied_xyz.shape[0]}")
        
        if occupied_xyz.shape[0] > 0:
            print(f"Occupancy range: {occupied_occ.min().item():.2f} - {occupied_occ.max().item():.2f}")
            xyz_np = occupied_xyz.cpu().numpy()
            print(f"Spatial range X: {xyz_np[:, 0].min():.2f} - {xyz_np[:, 0].max():.2f}m")
            print(f"Spatial range Y: {xyz_np[:, 1].min():.2f} - {xyz_np[:, 1].max():.2f}m")
            print(f"Spatial range Z: {xyz_np[:, 2].min():.2f} - {xyz_np[:, 2].max():.2f}m")
    
    # Test 2D projection
    print("\n--- Testing 2D projection ---")
    occupancy_2d = voxel_map.get_occupancy_map_2d(height_range=(0.5, 2.5))
    print(f"2D occupancy map shape: {occupancy_2d.shape}")
    print(f"Occupied pixels in 2D map: {np.sum(occupancy_2d > 0)}")
    
    # Test save/load functionality
    print("\n--- Testing save/load ---")
    test_file = '/tmp/test_voxel_map.pt'
    voxel_map.save_map(test_file)
    print(f"Saved map to {test_file}")
    
    # Create new map and load
    new_map = OccupancyVoxelMap(device=device)
    new_map.load_map(test_file)
    print(f"Loaded map with {new_map.get_voxel_count()} voxels")
    
    # Verify loaded map matches
    if voxel_map.get_voxel_count() == new_map.get_voxel_count():
        print("✓ Save/load test passed!")
    else:
        print("✗ Save/load test failed!")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\n--- Visualization Test ---")
    try:
        # Create a simple visualization
        if occupancy_2d.max() > 0:
            plt.figure(figsize=(10, 10))
            plt.imshow(occupancy_2d, cmap='viridis', origin='upper')
            plt.title('2D Occupancy Map Projection')
            plt.colorbar(label='Occupancy')
            plt.xlabel('X (pixels)')
            plt.ylabel('Y (pixels)')
            
            # Save visualization
            vis_file = '/tmp/occ_voxel_map_test.png'
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {vis_file}")
            plt.close()
        else:
            print("No occupied voxels to visualize")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("\n✓ OccupancyVoxelMap integration test completed successfully!")
    return voxel_map


if __name__ == "__main__":
    try:
        voxel_map = test_occ_voxel_map()
        print(f"\nFinal map statistics:")
        print(f"- Total voxels: {voxel_map.get_voxel_count()}")
        print(f"- Voxel size: {voxel_map.vox_size}m")
        print(f"- Device: {voxel_map.device}")
        print(f"- Is empty: {voxel_map.is_empty()}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
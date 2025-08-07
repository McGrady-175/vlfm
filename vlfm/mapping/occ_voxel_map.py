# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

"""Module defining an occupancy voxel map class for VLFM.

This is adapted from RayFronts' OccupancyVoxelMap to work with VLFM's architecture.
"""

from typing import Any, List, Tuple, Union, Optional
import torch
import numpy as np

from vlfm.mapping.base_map import BaseMap
from vlfm.mapping.geometry3d import depth_to_sparse_occupancy_voxels, pointcloud_to_sparse_voxels


class OccupancyVoxelMap(BaseMap):
    """A 3D occupancy voxel map using PyTorch tensors.
    
    This class maintains a sparse voxel representation where each voxel stores
    log-odds occupancy values. It processes depth images to update the occupancy
    state of the environment.
    
    Attributes:
        vox_size: The size of each voxel in meters.
        max_pts_per_frame: Maximum points to process per frame.
        max_empty_pts_per_frame: Maximum empty points to process per frame.
        max_depth_sensing: Maximum sensing range for depth.
        max_empty_cnt: Maximum log-odds value for empty voxels.
        max_occ_cnt: Maximum log-odds value for occupied voxels.
        occ_observ_weight: Weight for occupied observations.
        occ_thickness: Thickness of occupied voxels.
        device: Device for computation (cuda/cpu).
        
        global_vox_xyz: Nx3 tensor of voxel positions in world coordinates.
        global_vox_occ: Nx1 tensor of log-odds occupancy values.
    """
    
    def __init__(
        self,
        vox_size: float = 0.1,
        max_pts_per_frame: int = 1000,
        max_empty_pts_per_frame: int = 1000,
        max_depth_sensing: float = -1,
        max_empty_cnt: int = 3,
        max_occ_cnt: int = 5,
        occ_observ_weight: int = 5,
        occ_thickness: int = 2,
        vox_accum_period: int = 1,
        device: str = None,
        clip_bbox: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        size: int = 1000,
        pixels_per_meter: int = 20,
        *args: Any,
        **kwargs: Any
    ):
        """Initialize the occupancy voxel map.
        
        Args:
            vox_size: Length of a side of a voxel in meters.
            max_pts_per_frame: How many points to project per frame. Set to -1 to 
                project all valid depth points.
            max_empty_pts_per_frame: How many empty points to project per frame.
            max_depth_sensing: Maximum sensing range. Set to -1 to use max depth in frame.
            max_empty_cnt: Maximum log-odds value for empty voxels.
            max_occ_cnt: Maximum log-odds value for occupied voxels.
            occ_observ_weight: Weight for occupied observations vs empty.
            occ_thickness: Thickness of occupied surfaces in voxels.
            vox_accum_period: How often to aggregate voxels into global representation.
            device: Computation device (cuda/cpu).
            clip_bbox: Bounding box to limit mapping region ((min_x,min_y,min_z), (max_x,max_y,max_z)).
            size: Size of the 2D map for visualization (inherited from BaseMap).
            pixels_per_meter: Pixels per meter for visualization.
        """
        super().__init__(size, pixels_per_meter, *args, **kwargs)
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Voxel parameters
        self.vox_size = vox_size
        self.max_pts_per_frame = max_pts_per_frame
        self.max_empty_pts_per_frame = max_empty_pts_per_frame
        self.max_depth_sensing = max_depth_sensing
        self.max_empty_cnt = max_empty_cnt
        self.max_occ_cnt = max_occ_cnt
        self.occ_observ_weight = occ_observ_weight
        self.occ_thickness = occ_thickness
        
        # Accumulation parameters
        self.vox_accum_period = vox_accum_period
        self._vox_accum_cnt = 0
        self._tmp_vox_xyz = []
        self._tmp_vox_occ = []
        
        # Clipping bbox
        self.clip_bbox = None
        if clip_bbox is not None:
            self.clip_bbox = torch.tensor(clip_bbox, dtype=torch.float, device=self.device)
            assert self.clip_bbox.shape[0] == 2 and self.clip_bbox.shape[1] == 3
        
        # Global voxel representation
        self.global_vox_xyz = None
        self.global_vox_occ = None
        
    def reset(self) -> None:
        """Reset the map."""
        super().reset()
        self.global_vox_xyz = None
        self.global_vox_occ = None
        self._vox_accum_cnt = 0
        self._tmp_vox_xyz.clear()
        self._tmp_vox_occ.clear()
        
    def _clip_pc(self, pc_xyz: torch.Tensor, *features) -> List[torch.Tensor]:
        """Clip point cloud to bounding box if set."""
        if self.clip_bbox is None:
            return [pc_xyz] + list(features)
            
        mask = (pc_xyz > self.clip_bbox[0]) & (pc_xyz < self.clip_bbox[1])
        mask = torch.all(mask, dim=-1)
        pc_xyz = pc_xyz[mask]
        result = [pc_xyz]
        for f in features:
            result.append(f[mask])
        return result
        
    def update_map(
        self,
        depth: Union[np.ndarray, torch.Tensor],
        tf_camera_to_episodic: np.ndarray,
        intrinsics: np.ndarray,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        conf_map: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        """Update the occupancy map with a new depth observation.
        
        Args:
            depth: Depth image as numpy array or torch tensor.
            tf_camera_to_episodic: 4x4 transformation matrix from camera to world.
            intrinsics: 3x3 camera intrinsics matrix.
            min_depth: Minimum valid depth value.
            max_depth: Maximum valid depth value.
            conf_map: Optional confidence map for depth values.
        """
        # Convert inputs to torch tensors
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).float()
        if isinstance(tf_camera_to_episodic, np.ndarray):
            tf_camera_to_episodic = torch.from_numpy(tf_camera_to_episodic).float()
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics).float()
            
        # Move to device
        depth = depth.to(self.device)
        tf_camera_to_episodic = tf_camera_to_episodic.to(self.device)
        intrinsics = intrinsics.to(self.device)
        
        # Ensure depth has correct shape (Bx1xHxW)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(0)
        elif depth.dim() == 4:
            pass  # Already correct
        else:
            raise ValueError(f"Invalid depth shape: {depth.shape}")
            
        # Ensure pose has correct shape (Bx4x4)
        if tf_camera_to_episodic.dim() == 2:
            tf_camera_to_episodic = tf_camera_to_episodic.unsqueeze(0)
            
        # Process depth to voxels
        vox_xyz, vox_occ = depth_to_sparse_occupancy_voxels(
            depth, tf_camera_to_episodic, intrinsics, self.vox_size,
            conf_map=conf_map,
            max_num_pts=self.max_pts_per_frame,
            max_num_empty_pts=self.max_empty_pts_per_frame,
            max_depth_sensing=self.max_depth_sensing,
            occ_thickness=self.occ_thickness
        )
        
        # Clip to bounding box if specified
        vox_xyz, vox_occ = self._clip_pc(vox_xyz, vox_occ)
        
        # Convert [0, 1] occupancy to log-odds [-1, occ_observ_weight]
        vox_occ = vox_occ * self.occ_observ_weight - 1
        
        # Accumulate voxels
        B = depth.shape[0]
        self._vox_accum_cnt += B
        self._tmp_vox_occ.append(vox_occ)
        self._tmp_vox_xyz.append(vox_xyz)
        
        if self._vox_accum_cnt >= self.vox_accum_period:
            self._vox_accum_cnt = 0
            self.accum_occ_voxels()
            
    def accum_occ_voxels(self) -> None:
        """Accumulate temporarily gathered occupancy voxels into global map."""
        if len(self._tmp_vox_xyz) == 0:
            return
            
        # Include existing global voxels if they exist
        if self.global_vox_xyz is not None:
            self._tmp_vox_xyz.append(self.global_vox_xyz)
            self._tmp_vox_occ.append(self.global_vox_occ)
            
        # Concatenate all accumulated voxels
        pts_xyz = torch.cat(self._tmp_vox_xyz, dim=0)
        pts_occ = torch.cat(self._tmp_vox_occ, dim=0)
        
        # Clear temporary storage
        self._tmp_vox_occ.clear()
        self._tmp_vox_xyz.clear()
        
        # Aggregate into sparse voxel representation
        self.global_vox_xyz, self.global_vox_occ = pointcloud_to_sparse_voxels(
            pts_xyz, feat_pc=pts_occ, vox_size=self.vox_size, aggregation="sum"
        )
        
        # Clamp occupancy values
        self.global_vox_occ = torch.clamp(
            self.global_vox_occ,
            min=-self.max_empty_cnt,
            max=self.max_occ_cnt
        )
        
    def get_occupancy_map_2d(self, height_range: Tuple[float, float] = (-1.0, 2.0)) -> np.ndarray:
        """Get a 2D occupancy map by projecting voxels within height range.
        
        Args:
            height_range: (min_height, max_height) range to include voxels.
            
        Returns:
            2D occupancy map as numpy array.
        """
        if self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0:
            return self._map.copy()
            
        # Filter voxels by height
        z_coords = self.global_vox_xyz[:, 2]
        height_mask = (z_coords >= height_range[0]) & (z_coords <= height_range[1])
        
        if not height_mask.any():
            return self._map.copy()
            
        filtered_xyz = self.global_vox_xyz[height_mask]
        filtered_occ = self.global_vox_occ[height_mask]
        
        # Convert to numpy for processing
        filtered_xyz = filtered_xyz.cpu().numpy()
        filtered_occ = filtered_occ.cpu().numpy()
        
        # Project to 2D pixel coordinates
        xy_points = filtered_xyz[:, :2]  # Take x, y coordinates
        pixel_points = self._xy_to_px(xy_points)
        
        # Create occupancy map
        occupancy_map = self._map.copy()
        
        # Convert log-odds to probability and threshold
        probabilities = 1.0 / (1.0 + np.exp(-filtered_occ.flatten()))
        occupied_mask = probabilities > 0.5
        
        # Set occupied pixels
        valid_mask = (
            (pixel_points[:, 0] >= 0) & 
            (pixel_points[:, 0] < self.size) & 
            (pixel_points[:, 1] >= 0) & 
            (pixel_points[:, 1] < self.size)
        )
        
        valid_occupied = valid_mask & occupied_mask
        if valid_occupied.any():
            valid_pixels = pixel_points[valid_occupied]
            occupancy_map[valid_pixels[:, 0], valid_pixels[:, 1]] = 1.0
            
        return occupancy_map
        
    def get_voxel_count(self) -> int:
        """Get the total number of voxels in the map."""
        if self.global_vox_xyz is None:
            return 0
        return self.global_vox_xyz.shape[0]
        
    def get_occupied_voxels(self, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get voxels above occupancy threshold.
        
        Args:
            threshold: Log-odds threshold for occupied voxels.
            
        Returns:
            Tuple of (xyz_positions, occupancy_values) for occupied voxels.
        """
        if self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0:
            empty_tensor = torch.zeros((0, 3), device=self.device)
            empty_occ = torch.zeros((0, 1), device=self.device)
            return empty_tensor, empty_occ
            
        occupied_mask = self.global_vox_occ.flatten() > threshold
        return self.global_vox_xyz[occupied_mask], self.global_vox_occ[occupied_mask]
        
    def save_map(self, file_path: str) -> None:
        """Save the voxel map to file.
        
        Args:
            file_path: Path to save the map.
        """
        save_dict = {
            'global_vox_xyz': self.global_vox_xyz,
            'global_vox_occ': self.global_vox_occ,
            'vox_size': self.vox_size,
            'parameters': {
                'max_pts_per_frame': self.max_pts_per_frame,
                'max_empty_pts_per_frame': self.max_empty_pts_per_frame,
                'max_depth_sensing': self.max_depth_sensing,
                'max_empty_cnt': self.max_empty_cnt,
                'max_occ_cnt': self.max_occ_cnt,
                'occ_observ_weight': self.occ_observ_weight,
                'occ_thickness': self.occ_thickness,
            }
        }
        torch.save(save_dict, file_path)
        
    def load_map(self, file_path: str) -> None:
        """Load the voxel map from file.
        
        Args:
            file_path: Path to load the map from.
        """
        data = torch.load(file_path, map_location=self.device)
        self.global_vox_xyz = data['global_vox_xyz']
        self.global_vox_occ = data['global_vox_occ']
        if 'vox_size' in data:
            self.vox_size = data['vox_size']
        if 'parameters' in data:
            params = data['parameters']
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
    def is_empty(self) -> bool:
        """Check if the map is empty."""
        return self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0
# OccupancyVoxelMap Integration in VLFM

This document describes the integration of the `occ_voxel_map` functionality from [RayFronts](https://github.com/RayFronts/RayFronts) into the VLFM (Vision-Language Frontier Maps) framework.

## Overview

The `OccupancyVoxelMap` provides a 3D sparse voxel-based occupancy mapping system that:

- Maintains log-odds occupancy values for each voxel
- Processes depth images to update occupancy state
- Uses PyTorch for efficient GPU acceleration
- Supports both 3D voxel representation and 2D projection
- Integrates seamlessly with VLFM's existing mapping infrastructure

## Key Features

### 3D Sparse Voxel Representation
- **Sparse Storage**: Only occupied or observed voxels are stored, making it memory efficient
- **Log-odds Occupancy**: Each voxel stores log-odds values for probabilistic occupancy reasoning
- **Configurable Resolution**: Voxel size can be adjusted based on application requirements

### Depth Image Processing
- **Real-time Updates**: Processes depth images to update voxel occupancy
- **Camera Pose Integration**: Uses camera poses to transform observations to world coordinates
- **Confidence Mapping**: Optional confidence weighting for depth observations

### Integration with VLFM
- **BaseMap Inheritance**: Extends VLFM's `BaseMap` for consistent interface
- **2D Projection**: Provides 2D occupancy maps for compatibility with existing VLFM components
- **Coordinate System**: Works with VLFM's episodic coordinate system

## Files Added

### Core Implementation
- `vlfm/mapping/geometry3d.py`: Geometric functions for voxel operations
- `vlfm/mapping/occ_voxel_map.py`: Main `OccupancyVoxelMap` class
- `vlfm/mapping/__init__.py`: Module exports

### Testing and Examples
- `vlfm/examples/test_occ_voxel_map.py`: Test script demonstrating usage
- `vlfm/mapping/README_occ_voxel_map.md`: This documentation

## Usage Example

```python
from vlfm.mapping.occ_voxel_map import OccupancyVoxelMap
import torch
import numpy as np

# Create the occupancy voxel map
voxel_map = OccupancyVoxelMap(
    vox_size=0.1,  # 10cm voxels
    max_pts_per_frame=5000,
    device='cuda',  # Use GPU acceleration
    clip_bbox=((-10, -10, -2), (10, 10, 5))  # Limit mapping region
)

# Update with depth observation
depth = np.array(...)  # Your depth image (H, W)
camera_pose = np.eye(4)  # 4x4 transformation matrix
intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

voxel_map.update_map(
    depth=depth,
    tf_camera_to_episodic=camera_pose,
    intrinsics=intrinsics,
    min_depth=0.1,
    max_depth=10.0
)

# Get 2D occupancy map for visualization/planning
occupancy_2d = voxel_map.get_occupancy_map_2d(height_range=(0.5, 2.5))

# Get 3D occupied voxels
occupied_xyz, occupancy_values = voxel_map.get_occupied_voxels(threshold=0.0)

# Save and load maps
voxel_map.save_map("my_map.pt")
new_map = OccupancyVoxelMap()
new_map.load_map("my_map.pt")
```

## Configuration Parameters

### Voxel Parameters
- `vox_size`: Size of each voxel in meters (default: 0.1)
- `max_pts_per_frame`: Maximum points to process per frame (default: 1000)
- `max_empty_pts_per_frame`: Maximum empty points per frame (default: 1000)

### Occupancy Parameters
- `max_empty_cnt`: Maximum log-odds for empty voxels (default: 3)
- `max_occ_cnt`: Maximum log-odds for occupied voxels (default: 5)
- `occ_observ_weight`: Weight for occupied observations (default: 5)
- `occ_thickness`: Thickness of occupied surfaces in voxels (default: 2)

### Performance Parameters
- `vox_accum_period`: How often to aggregate voxels (default: 1)
- `device`: Computation device ('cuda' or 'cpu')
- `clip_bbox`: Bounding box to limit mapping region

## Key Methods

### Map Updates
- `update_map()`: Process new depth observation
- `accum_occ_voxels()`: Aggregate temporary voxels into global map
- `reset()`: Clear the map

### Data Access
- `get_occupancy_map_2d()`: Get 2D projection of occupancy map
- `get_occupied_voxels()`: Get 3D voxels above threshold
- `get_voxel_count()`: Get total number of voxels
- `is_empty()`: Check if map is empty

### Persistence
- `save_map()`: Save map to file
- `load_map()`: Load map from file

## Integration Benefits

### Memory Efficiency
- Sparse representation only stores observed voxels
- Configurable accumulation to balance memory and latency
- Efficient PyTorch tensor operations

### Real-time Performance
- GPU acceleration support
- Optimized voxelization algorithms
- Configurable point sampling for performance tuning

### VLFM Compatibility
- Extends existing `BaseMap` interface
- Provides 2D projections for existing VLFM components
- Compatible with VLFM's coordinate systems and transformations

### Extensibility
- Modular design allows easy customization
- Support for confidence weighting
- Configurable bounding boxes for region of interest

## Dependencies

The implementation requires:
- PyTorch (for tensor operations)
- NumPy (for numerical computations)
- Matplotlib (for visualization in examples)

## Future Enhancements

Potential improvements that could be added:
- Integration with torch_scatter for more efficient aggregation
- Support for semantic voxels (per-class occupancy)
- Dynamic voxel resolution based on distance
- Integration with frontier detection
- Multi-scale voxel representations

## Original Reference

This implementation is based on the `OccupancyVoxelMap` from the RayFronts project:
- Repository: https://github.com/RayFronts/RayFronts
- Paper: "RayFronts: Open-Set Semantic Ray Frontiers for Online Scene Understanding and Exploration"

The code has been adapted to work within VLFM's architecture while maintaining the core functionality and efficiency of the original implementation.
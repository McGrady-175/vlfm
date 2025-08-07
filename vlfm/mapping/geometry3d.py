"""Functions related to geometrical manipulation for voxel mapping."""

from typing import Tuple
import torch
import numpy as np

def pts_to_homogen(pts):
    """Convert points to homogeneous coordinates."""
    if pts.shape[-1] != 3:
        raise ValueError(f"Invalid points tensor shape {pts.shape}. "
                        "Last dim should have length 3")
    return torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)

def pts_to_nonhomo(pts):
    """Convert homogeneous points to non-homogeneous coordinates."""
    if pts.shape[-1] != 4:
        raise ValueError(f"Invalid points tensor shape {pts.shape}. "
                        "Last dim should have length 4")
    return pts[..., :3]

def mat_3x4_to_4x4(mat):
    """Convert 3x4 matrix to 4x4 matrix."""
    row = torch.tensor([[0,0,0,1]], device=mat.device)
    mat = torch.cat((mat, row.repeat(*mat.shape[:-2], 1, 1)), axis=-2)
    return mat

def mat_3x3_to_4x4(mat):
    """Convert 3x3 matrix to 4x4 matrix."""
    zeros = torch.zeros(size=(*mat.shape[:-2], 3, 1), device=mat.device)
    mat = torch.cat((mat, zeros), dim=-1)
    return mat_3x4_to_4x4(mat)

def transform_points_homo(homo_points, transform_mat_4x4):
    """Transform homogeneous points using 4x4 transformation matrix."""
    return homo_points @ torch.transpose(transform_mat_4x4, -2, -1)

def transform_points(points, transform_mat):
    """Transform 3D points using transformation matrix."""
    # Input checks and conditioning
    if transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 3:
        transform_mat = mat_3x3_to_4x4(transform_mat)
    elif transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 4:
        transform_mat = mat_3x4_to_4x4(transform_mat)
    
    if points.shape[-1] == 3:
        points = pts_to_homogen(points)
    
    transformed = transform_points_homo(points, transform_mat)
    return pts_to_nonhomo(transformed)

def pointcloud_to_sparse_voxels(xyz_pc, vox_size, feat_pc=None, aggregation="mean", return_counts=False):
    """Convert a point cloud to sparse voxels with feature aggregation.
    
    Args:
        xyz_pc: Nx3 float tensor including the xyz positions of each point.
        vox_size: Size of a voxel in world units.
        feat_pc: NxC float tensor including any features that will be aggregated
          through voxelization.
        aggregation: ['mean', 'sum']
        return_counts: Whether to return the number of points aggregated within each
        voxel or not.

    Returns:
        xyz_vx: Nx3 float tensor representing xyz centers of the voxels.
        feat_vx: (Returned if feat_pc is not None) NxC float tensor representing the
          aggregated features in each voxel
        count_vx: (Returned if return_counts is True) Nx1 float tensor representing
          how many points were aggregated in each voxel.
    """
    d = xyz_pc.device
    
    # Quantize points to voxel grid
    xyz_vx = torch.round(xyz_pc/vox_size).type(torch.int64)
    
    if feat_pc is None:
        xyz_vx, count_vx = torch.unique(xyz_vx, return_counts=True, dim=0)
        xyz_vx = xyz_vx.type(torch.float)*vox_size
        count_vx = count_vx.type(torch.float).unsqueeze(-1)
        if return_counts:
            return xyz_vx, count_vx
        else:
            return xyz_vx
    
    # Handle feature aggregation
    xyz_vx, reduce_ind, counts_vx = torch.unique(xyz_vx, return_inverse=True,
                                                return_counts=True, dim=0)
    feat_vx = torch.zeros((xyz_vx.shape[0], feat_pc.shape[-1]), device=d,
                         dtype=feat_pc.dtype)
    
    # Use simple aggregation instead of torch_scatter for now
    for i in range(feat_vx.shape[0]):
        mask = reduce_ind == i
        if aggregation == "mean":
            feat_vx[i] = feat_pc[mask].mean(dim=0)
        elif aggregation == "sum":
            feat_vx[i] = feat_pc[mask].sum(dim=0)
    
    xyz_vx = xyz_vx.type(torch.float)*vox_size
    counts_vx = counts_vx.type(torch.float).unsqueeze(-1)
    
    if return_counts:
        return xyz_vx, feat_vx, counts_vx
    else:
        return xyz_vx, feat_vx

def depth_to_sparse_occupancy_voxels(depth_img: torch.FloatTensor,
                                    pose_4x4: torch.FloatTensor,
                                    intrinsics_3x3: torch.FloatTensor,
                                    vox_size: float,
                                    conf_map: torch.FloatTensor = None,
                                    max_num_pts: int = -1,
                                    max_num_empty_pts: int = -1,
                                    max_depth_sensing: float = -1,
                                    occ_thickness: int = 1):
    """Convert depth image to sparse occupancy voxels.
    
    Args:
        depth_img: A Bx1xHxW float tensor with depth values.
        pose_4x4: A Bx4x4 tensor with camera poses.
        intrinsics_3x3: A 3x3 float tensor with camera intrinsics.
        vox_size: The voxel size of the voxel grid.
        conf_map: Optional confidence map.
        max_num_pts: Maximum number of points to project per image.
        max_num_empty_pts: Maximum number of empty points to project.
        max_depth_sensing: Maximum sensing range.
        occ_thickness: Thickness of occupied voxels.
        
    Returns:
        xyz_vx: Nx3 float tensor representing xyz centers of the voxels.
        occ_vx: Nx1 float tensor representing occupancy values (0=empty, 1=occupied).
    """
    B, _, H, W = depth_img.shape
    device = depth_img.device
    valid_depth_mask = torch.logical_and(torch.isfinite(depth_img), depth_img > 0)
    
    # Create image plane points
    img_xi, img_yi = torch.meshgrid(torch.arange(W, device=device),
                                   torch.arange(H, device=device),
                                   indexing="xy")
    img_xi = img_xi.tile((B, 1, 1))
    img_yi = img_yi.tile((B, 1, 1))
    
    img_plane_pts = torch.stack([
        img_xi.flatten(-2),
        img_yi.flatten(-2),
        torch.ones_like(img_xi.flatten(-2))
    ], dim=-1)
    
    # Project to 3D points
    depth_flat = depth_img.reshape(B, H*W, 1)
    pts_3d_cam = img_plane_pts * depth_flat
    
    # Transform camera intrinsics
    inv_intrinsics = torch.inverse(intrinsics_3x3)
    pts_3d_cam = pts_3d_cam @ inv_intrinsics.T
    
    # Transform to world coordinates
    pts_3d_world = transform_points(pts_3d_cam, pose_4x4)
    
    # Filter valid points
    valid_mask = valid_depth_mask.reshape(B, H*W)
    valid_pts = []
    
    for b in range(B):
        if valid_mask[b].any():
            valid_indices = torch.where(valid_mask[b])[0]
            if max_num_pts > 0 and len(valid_indices) > max_num_pts:
                # Sample points
                perm = torch.randperm(len(valid_indices), device=device)
                valid_indices = valid_indices[perm[:max_num_pts]]
            valid_pts.append(pts_3d_world[b, valid_indices])
    
    if not valid_pts:
        # Return empty voxels
        empty_xyz = torch.zeros((0, 3), device=device)
        empty_occ = torch.zeros((0, 1), device=device)
        return empty_xyz, empty_occ
    
    # Combine all valid points
    all_points = torch.cat(valid_pts, dim=0)
    
    # Create occupancy labels (all occupied points)
    occupancy_labels = torch.ones((all_points.shape[0], 1), device=device)
    
    # Voxelize
    xyz_vx, occ_vx = pointcloud_to_sparse_voxels(
        all_points, vox_size, feat_pc=occupancy_labels, aggregation="sum"
    )
    
    # Clamp occupancy to [0, 1]
    occ_vx = torch.clamp(occ_vx, max=1)
    
    return xyz_vx, occ_vx
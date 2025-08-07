# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Tuple, Optional, Union
import numpy as np
import cv2
from numba import jit
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import get_point_cloud, transform_points

try:
    import pyopenvdb as vdb
    OPENVDB_AVAILABLE = True
except ImportError:
    print("OpenVDB not available. Install with: pip install pyopenvdb")
    OPENVDB_AVAILABLE = False


class OccupancyVoxelMap(BaseMap):
    """
    3D占用体素地图实现，基于OpenVDB的高性能体素化表示
    支持实时更新、多线程处理和内存优化
    """
    
    def __init__(
        self,
        voxel_size: float = 0.1,  # 体素大小（米）
        min_height: float = -2.0,  # 最低高度
        max_height: float = 3.0,   # 最高高度
        prob_hit: float = 0.9,     # 命中概率
        prob_miss: float = 0.4,    # 未命中概率
        occupancy_threshold: float = 0.5,  # 占用阈值
        max_range: float = 10.0,   # 最大感知距离
        num_threads: int = 4,      # 线程数
        enable_optimizations: bool = True,  # 启用优化
        **kwargs
    ):
        if not OPENVDB_AVAILABLE:
            raise ImportError("OpenVDB is required for VoxelMap")
            
        super().__init__(**kwargs)
        
        # 体素地图参数
        self.voxel_size = voxel_size
        self.min_height = min_height
        self.max_height = max_height
        self.prob_hit = prob_hit
        self.prob_miss = prob_miss
        self.occupancy_threshold = occupancy_threshold
        self.max_range = max_range
        self.num_threads = num_threads
        self.enable_optimizations = enable_optimizations
        
        # 对数几率表示
        self.log_odds_hit = np.log(prob_hit / (1 - prob_hit))
        self.log_odds_miss = np.log(prob_miss / (1 - prob_miss))
        
        # OpenVDB网格
        self.occupancy_grid = vdb.FloatGrid()
        self.occupancy_grid.name = "occupancy"
        self.occupancy_grid.background = 0.0
        
        # 探索区域网格
        self.explored_grid = vdb.BoolGrid()
        self.explored_grid.name = "explored"
        self.explored_grid.background = False
        
        # 距离场网格（可选）
        self.distance_grid = vdb.FloatGrid()
        self.distance_grid.name = "distance"
        self.distance_grid.background = float('inf')
        
        # 线程锁
        self.update_lock = threading.RLock()
        
        # 统计信息
        self.total_updates = 0
        self.update_times = []
        
    def reset(self) -> None:
        """重置地图"""
        super().reset()
        with self.update_lock:
            self.occupancy_grid.clear()
            self.explored_grid.clear()
            self.distance_grid.clear()
            self.total_updates = 0
            self.update_times.clear()
    
    def world_to_voxel(self, world_coords: np.ndarray) -> np.ndarray:
        """世界坐标转体素坐标"""
        return np.floor(world_coords / self.voxel_size).astype(np.int32)
    
    def voxel_to_world(self, voxel_coords: np.ndarray) -> np.ndarray:
        """体素坐标转世界坐标"""
        return (voxel_coords + 0.5) * self.voxel_size
    
    def update_map(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        fov: float = None,
        subsample_factor: int = 1,
    ) -> None:
        """
        更新3D体素地图
        
        Args:
            depth: 深度图像 (H, W)
            tf_camera_to_episodic: 相机到世界坐标系的变换矩阵
            min_depth: 最小深度值（米）
            max_depth: 最大深度值（米）
            fx, fy: 相机内参焦距
            fov: 视场角（可选）
            subsample_factor: 子采样因子（用于性能优化）
        """
        start_time = time.time()
        
        # 生成点云
        scaled_depth = depth * (max_depth - min_depth) + min_depth
        mask = (scaled_depth > min_depth) & (scaled_depth < self.max_range)
        
        # 子采样以提高性能
        if subsample_factor > 1:
            mask = mask[::subsample_factor, ::subsample_factor]
            scaled_depth = scaled_depth[::subsample_factor, ::subsample_factor]
        
        point_cloud_camera = get_point_cloud(scaled_depth, mask, fx, fy)
        point_cloud_world = transform_points(tf_camera_to_episodic, point_cloud_camera)
        
        # 过滤高度范围
        height_mask = (point_cloud_world[:, 2] >= self.min_height) & \
                     (point_cloud_world[:, 2] <= self.max_height)
        valid_points = point_cloud_world[height_mask]
        
        # 获取传感器位置
        sensor_pos = tf_camera_to_episodic[:3, 3]
        
        # 更新体素地图
        self._update_voxels_batch(sensor_pos, valid_points)
        
        # 记录更新时间
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        self.total_updates += 1
        
        # 可选：更新距离场
        if self.enable_optimizations:
            self._update_distance_field()
    
    def _update_voxels_batch(self, sensor_pos: np.ndarray, hit_points: np.ndarray) -> None:
        """批量更新体素"""
        if len(hit_points) == 0:
            return
            
        with self.update_lock:
            # 使用多线程处理射线投射
            if self.num_threads > 1 and len(hit_points) > 100:
                self._parallel_raycast(sensor_pos, hit_points)
            else:
                self._sequential_raycast(sensor_pos, hit_points)
    
    def _sequential_raycast(self, sensor_pos: np.ndarray, hit_points: np.ndarray) -> None:
        """顺序射线投射"""
        sensor_voxel = self.world_to_voxel(sensor_pos.reshape(1, -1))[0]
        
        for hit_point in hit_points:
            hit_voxel = self.world_to_voxel(hit_point.reshape(1, -1))[0]
            ray_voxels = self._bresenham_3d(sensor_voxel, hit_voxel)
            
            # 更新射线路径上的体素（空闲）
            for i, voxel in enumerate(ray_voxels[:-1]):
                coord = vdb.Coord(*voxel.astype(int))
                current_value = self.occupancy_grid.evalActiveVoxel(coord)
                new_value = current_value + self.log_odds_miss
                self.occupancy_grid.setActiveVoxel(coord, new_value)
                self.explored_grid.setActiveVoxel(coord, True)
            
            # 更新终点体素（占用）
            if len(ray_voxels) > 0:
                hit_coord = vdb.Coord(*hit_voxel.astype(int))
                current_value = self.occupancy_grid.evalActiveVoxel(hit_coord)
                new_value = current_value + self.log_odds_hit
                self.occupancy_grid.setActiveVoxel(hit_coord, new_value)
                self.explored_grid.setActiveVoxel(hit_coord, True)
    
    def _parallel_raycast(self, sensor_pos: np.ndarray, hit_points: np.ndarray) -> None:
        """并行射线投射"""
        chunk_size = max(1, len(hit_points) // self.num_threads)
        chunks = [hit_points[i:i + chunk_size] for i in range(0, len(hit_points), chunk_size)]
        
        # 为每个线程创建临时网格
        temp_grids = []
        temp_explored = []
        
        def process_chunk(chunk):
            temp_occ = vdb.FloatGrid()
            temp_exp = vdb.BoolGrid()
            
            sensor_voxel = self.world_to_voxel(sensor_pos.reshape(1, -1))[0]
            
            for hit_point in chunk:
                hit_voxel = self.world_to_voxel(hit_point.reshape(1, -1))[0]
                ray_voxels = self._bresenham_3d(sensor_voxel, hit_voxel)
                
                # 更新临时网格
                for voxel in ray_voxels[:-1]:
                    coord = vdb.Coord(*voxel.astype(int))
                    current_value = temp_occ.evalActiveVoxel(coord)
                    temp_occ.setActiveVoxel(coord, current_value + self.log_odds_miss)
                    temp_exp.setActiveVoxel(coord, True)
                
                if len(ray_voxels) > 0:
                    hit_coord = vdb.Coord(*hit_voxel.astype(int))
                    current_value = temp_occ.evalActiveVoxel(hit_coord)
                    temp_occ.setActiveVoxel(hit_coord, current_value + self.log_odds_hit)
                    temp_exp.setActiveVoxel(hit_coord, True)
            
            return temp_occ, temp_exp
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(process_chunk, chunks))
        
        # 合并结果
        for temp_occ, temp_exp in results:
            self.occupancy_grid.combineWith(temp_occ, vdb.CSG_UNION_A)
            self.explored_grid.combineWith(temp_exp, vdb.CSG_UNION_A)
    
    @staticmethod
    @jit(nopython=True)
    def _bresenham_3d(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """3D Bresenham算法生成射线路径"""
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        dz = abs(end[2] - start[2])
        
        sx = 1 if start[0] < end[0] else -1
        sy = 1 if start[1] < end[1] else -1
        sz = 1 if start[2] < end[2] else -1
        
        dm = max(dx, dy, dz)
        voxels = []
        
        x, y, z = start[0], start[1], start[2]
        voxels.append(np.array([x, y, z]))
        
        if dm == dx:
            e1, e2 = dy - dx, dz - dx
            for _ in range(dx):
                x += sx
                if e1 >= 0:
                    y += sy
                    e1 -= dx
                if e2 >= 0:
                    z += sz
                    e2 -= dx
                e1 += dy
                e2 += dz
                voxels.append(np.array([x, y, z]))
        elif dm == dy:
            e1, e2 = dx - dy, dz - dy
            for _ in range(dy):
                y += sy
                if e1 >= 0:
                    x += sx
                    e1 -= dy
                if e2 >= 0:
                    z += sz
                    e2 -= dy
                e1 += dx
                e2 += dz
                voxels.append(np.array([x, y, z]))
        else:
            e1, e2 = dx - dz, dy - dz
            for _ in range(dz):
                z += sz
                if e1 >= 0:
                    x += sx
                    e1 -= dz
                if e2 >= 0:
                    y += sy
                    e2 -= dz
                e1 += dx
                e2 += dy
                voxels.append(np.array([x, y, z]))
        
        return np.array(voxels)
    
    def _update_distance_field(self) -> None:
        """更新距离场（用于路径规划优化）"""
        # 这里可以实现TSDF或ESDF更新
        # 暂时跳过，因为这是一个复杂的操作
        pass
    
    def is_occupied(self, world_pos: np.ndarray) -> bool:
        """检查世界坐标点是否被占用"""
        voxel_coord = self.world_to_voxel(world_pos.reshape(1, -1))[0]
        coord = vdb.Coord(*voxel_coord.astype(int))
        log_odds = self.occupancy_grid.evalActiveVoxel(coord)
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        return probability > self.occupancy_threshold
    
    def is_free(self, world_pos: np.ndarray) -> bool:
        """检查世界坐标点是否为自由空间"""
        voxel_coord = self.world_to_voxel(world_pos.reshape(1, -1))[0]
        coord = vdb.Coord(*voxel_coord.astype(int))
        
        # 检查是否已探索
        if not self.explored_grid.evalActiveVoxel(coord):
            return False  # 未探索区域
        
        log_odds = self.occupancy_grid.evalActiveVoxel(coord)
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        return probability < (1.0 - self.occupancy_threshold)
    
    def get_occupancy_probability(self, world_pos: np.ndarray) -> float:
        """获取世界坐标点的占用概率"""
        voxel_coord = self.world_to_voxel(world_pos.reshape(1, -1))[0]
        coord = vdb.Coord(*voxel_coord.astype(int))
        log_odds = self.occupancy_grid.evalActiveVoxel(coord)
        return 1.0 / (1.0 + np.exp(-log_odds))
    
    def get_occupied_voxels(self) -> np.ndarray:
        """获取所有占用的体素世界坐标"""
        occupied_coords = []
        accessor = self.occupancy_grid.getAccessor()
        
        for coord, value in self.occupancy_grid.iterOnVoxels():
            probability = 1.0 / (1.0 + np.exp(-value))
            if probability > self.occupancy_threshold:
                world_pos = self.voxel_to_world(np.array([coord.x, coord.y, coord.z]))
                occupied_coords.append(world_pos)
        
        return np.array(occupied_coords) if occupied_coords else np.empty((0, 3))
    
    def get_2d_projection(self, height_range: Tuple[float, float] = None) -> np.ndarray:
        """获取2D投影地图（用于与现有VLFM系统兼容）"""
        if height_range is None:
            height_range = (self.min_height, self.max_height)
        
        # 创建2D网格
        projection_map = np.zeros((self.size, self.size), dtype=np.float32)
        
        # 遍历体素网格
        for coord, value in self.occupancy_grid.iterOnVoxels():
            world_pos = self.voxel_to_world(np.array([coord.x, coord.y, coord.z]))
            
            # 检查高度范围
            if height_range[0] <= world_pos[2] <= height_range[1]:
                probability = 1.0 / (1.0 + np.exp(-value))
                
                # 转换到像素坐标
                pixel_coords = self._xy_to_px(world_pos[:2].reshape(1, -1))[0]
                px, py = pixel_coords
                
                if 0 <= px < self.size and 0 <= py < self.size:
                    # 取最大占用概率
                    projection_map[px, py] = max(projection_map[px, py], probability)
        
        return projection_map
    
    def visualize(self, show_explored: bool = False) -> np.ndarray:
        """可视化体素地图"""
        projection = self.get_2d_projection()
        
        # 转换为彩色图像
        visualization = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # 占用区域（红色）
        occupied_mask = projection > self.occupancy_threshold
        visualization[occupied_mask] = [255, 0, 0]
        
        # 自由区域（绿色）
        free_mask = (projection > 0) & (projection <= (1.0 - self.occupancy_threshold))
        visualization[free_mask] = [0, 255, 0]
        
        # 未知区域（灰色）
        unknown_mask = projection == 0
        visualization[unknown_mask] = [128, 128, 128]
        
        return visualization
    
    def get_statistics(self) -> dict:
        """获取地图统计信息"""
        total_voxels = self.occupancy_grid.activeVoxelCount()
        occupied_count = 0
        free_count = 0
        
        for coord, value in self.occupancy_grid.iterOnVoxels():
            probability = 1.0 / (1.0 + np.exp(-value))
            if probability > self.occupancy_threshold:
                occupied_count += 1
            elif probability < (1.0 - self.occupancy_threshold):
                free_count += 1
        
        avg_update_time = np.mean(self.update_times) if self.update_times else 0.0
        
        return {
            "total_voxels": total_voxels,
            "occupied_voxels": occupied_count,
            "free_voxels": free_count,
            "unknown_voxels": total_voxels - occupied_count - free_count,
            "total_updates": self.total_updates,
            "avg_update_time": avg_update_time,
            "voxel_size": self.voxel_size,
            "memory_usage_mb": self._estimate_memory_usage(),
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        # 简单估算：每个活跃体素约16字节（占用概率 + 探索状态）
        active_voxels = self.occupancy_grid.activeVoxelCount()
        return (active_voxels * 16) / (1024 * 1024)
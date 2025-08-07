# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.mapping.voxel_map import OccupancyVoxelMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class VoxelNavPolicy(BaseObjectNavPolicy):
    """基于3D体素地图的导航策略"""
    
    def __init__(
        self,
        # 体素地图参数
        voxel_size: float = 0.1,
        voxel_min_height: float = -2.0,
        voxel_max_height: float = 3.0,
        voxel_max_range: float = 10.0,
        enable_3d_planning: bool = True,
        planning_height_clearance: float = 0.5,
        # 其他参数
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self._voxel_size = voxel_size
        self._voxel_min_height = voxel_min_height
        self._voxel_max_height = voxel_max_height
        self._voxel_max_range = voxel_max_range
        self._enable_3d_planning = enable_3d_planning
        self._planning_height_clearance = planning_height_clearance
        
        # 初始化3D体素地图
        self._voxel_map = OccupancyVoxelMap(
            voxel_size=voxel_size,
            min_height=voxel_min_height,
            max_height=voxel_max_height,
            max_range=voxel_max_range,
            size=self._map_size_in_pixels,
            pixels_per_meter=self._pixels_per_meter,
        )
        
        # 2D投影地图（用于兼容性）
        self._projected_obstacle_map = None
        
    def reset(self) -> None:
        super().reset()
        self._voxel_map.reset()
        self._projected_obstacle_map = None
    
    def _get_observations_cache(self, observations: "TensorDict") -> None:
        """缓存观测数据并更新3D体素地图"""
        super()._get_observations_cache(observations)
        
        # 更新3D体素地图
        self._update_voxel_map()
        
        # 生成2D投影地图以保持兼容性
        self._update_projected_map()
    
    def _update_voxel_map(self) -> None:
        """更新3D体素地图"""
        if not self._observations_cache.get("nav_depth") is not None:
            return
            
        # 获取深度图像和相机参数
        depth = self._observations_cache["nav_depth"]
        camera_info = self._observations_cache.get("nav_camera_info", {})
        
        # 默认相机参数（如果没有提供）
        fx = camera_info.get("fx", depth.shape[1] / 2.0)
        fy = camera_info.get("fy", depth.shape[0] / 2.0)
        min_depth = camera_info.get("min_depth", 0.1)
        max_depth = camera_info.get("max_depth", 10.0)
        
        # 获取相机位姿
        tf_camera_to_episodic = self._observations_cache["camera_tf"]
        
        # 更新体素地图
        self._voxel_map.update_map(
            depth=depth,
            tf_camera_to_episodic=tf_camera_to_episodic,
            min_depth=min_depth,
            max_depth=max_depth,
            fx=fx,
            fy=fy,
            subsample_factor=2,  # 性能优化
        )
    
    def _update_projected_map(self) -> None:
        """更新2D投影地图以保持与现有系统的兼容性"""
        # 获取机器人高度范围的投影
        robot_height = self._observations_cache.get("robot_xy", np.array([0, 0, 0]))[2] if len(self._observations_cache.get("robot_xy", [])) > 2 else 0
        height_range = (
            robot_height + self._obstacle_map._min_height,
            robot_height + self._obstacle_map._max_height
        )
        
        self._projected_obstacle_map = self._voxel_map.get_2d_projection(height_range)
        
        # 更新原有的obstacle_map以保持兼容性
        self._obstacle_map._map = (self._projected_obstacle_map > self._voxel_map.occupancy_threshold)
    
    def _check_collision_3d(self, path: np.ndarray, robot_radius: float = 0.5) -> bool:
        """检查3D路径是否存在碰撞"""
        if not self._enable_3d_planning:
            return super()._check_collision_2d(path[:, :2], robot_radius)
        
        # 沿路径采样点进行碰撞检测
        for point in path:
            # 检查机器人周围的体素
            check_positions = self._generate_robot_volume_points(point, robot_radius)
            
            for pos in check_positions:
                if self._voxel_map.is_occupied(pos):
                    return True
        
        return False
    
    def _generate_robot_volume_points(self, center: np.ndarray, radius: float) -> np.ndarray:
        """生成机器人体积范围内的检查点"""
        points = []
        
        # 在机器人半径范围内生成采样点
        step = self._voxel_size / 2
        for dx in np.arange(-radius, radius + step, step):
            for dy in np.arange(-radius, radius + step, step):
                for dz in np.arange(-self._planning_height_clearance, 
                                   self._planning_height_clearance + step, step):
                    if dx*dx + dy*dy <= radius*radius:  # 圆形截面
                        point = center + np.array([dx, dy, dz])
                        points.append(point)
        
        return np.array(points)
    
    def _get_navigable_map_3d(self) -> np.ndarray:
        """获取3D可导航地图"""
        if not self._enable_3d_planning:
            return self._obstacle_map._navigable_map
        
        # 基于3D体素地图生成可导航区域
        navigable_map = np.ones_like(self._projected_obstacle_map, dtype=bool)
        
        # 将占用区域标记为不可导航
        occupied_mask = self._projected_obstacle_map > self._voxel_map.occupancy_threshold
        navigable_map &= ~occupied_mask
        
        # 考虑机器人半径进行膨胀
        if hasattr(self._obstacle_map, '_navigable_kernel'):
            navigable_map = cv2.erode(
                navigable_map.astype(np.uint8),
                self._obstacle_map._navigable_kernel,
                iterations=1
            ).astype(bool)
        
        return navigable_map
    
    def _plan_path_3d(self, start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, bool]:
        """3D路径规划"""
        if not self._enable_3d_planning:
            # 回退到2D规划
            path_2d, success = self._plan_path_2d(start[:2], goal[:2])
            if success:
                # 为2D路径添加高度信息
                path_3d = np.column_stack([
                    path_2d,
                    np.full(len(path_2d), start[2])  # 保持当前高度
                ])
                return path_3d, True
            return np.array([]), False
        
        # 使用A*算法进行3D路径规划
        return self._astar_3d(start, goal)
    
    def _astar_3d(self, start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, bool]:
        """3D A*路径规划算法"""
        # 简化的3D A*实现
        # 实际实现中可以使用更复杂的启发式函数和优化
        
        start_voxel = self._voxel_map.world_to_voxel(start.reshape(1, -1))[0]
        goal_voxel = self._voxel_map.world_to_voxel(goal.reshape(1, -1))[0]
        
        # 如果起点或终点被占用，返回失败
        if (self._voxel_map.is_occupied(start) or 
            self._voxel_map.is_occupied(goal)):
            return np.array([]), False
        
        # 简化实现：使用直线路径并检查碰撞
        direction = goal - start
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.array([start]), True
        
        direction_normalized = direction / distance
        
        # 沿直线采样点
        num_steps = int(distance / self._voxel_size) + 1
        path_points = []
        
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            point = start + t * direction
            
            # 检查碰撞
            if not self._voxel_map.is_occupied(point):
                path_points.append(point)
            else:
                # 遇到障碍物，尝试绕行（简化处理）
                if len(path_points) > 0:
                    return np.array(path_points), False
                else:
                    return np.array([]), False
        
        path_points.append(goal)
        return np.array(path_points), True
    
    def _get_frontier_points_3d(self) -> np.ndarray:
        """获取3D前沿点"""
        if not self._enable_3d_planning:
            return super()._get_frontier_points()
        
        # 基于3D体素地图生成前沿点
        frontiers_2d = super()._get_frontier_points()
        
        if len(frontiers_2d) == 0:
            return np.array([])
        
        # 为2D前沿点添加合适的高度信息
        frontiers_3d = []
        robot_height = self._observations_cache.get("robot_xy", np.array([0, 0, 0]))[2] if len(self._observations_cache.get("robot_xy", [])) > 2 else 0
        
        for frontier_2d in frontiers_2d:
            # 在该位置寻找合适的3D前沿点
            for height in np.arange(self._voxel_min_height, self._voxel_max_height, self._voxel_size):
                frontier_3d = np.array([frontier_2d[0], frontier_2d[1], robot_height + height])
                if self._voxel_map.is_free(frontier_3d):
                    frontiers_3d.append(frontier_3d)
                    break
        
        return np.array(frontiers_3d) if frontiers_3d else np.array([])
    
    def _visualize_voxel_map(self) -> np.ndarray:
        """可视化体素地图"""
        return self._voxel_map.visualize()
    
    def _get_policy_info(self, observations: "TensorDict") -> Dict[str, Any]:
        """获取策略信息，包括体素地图可视化"""
        policy_info = super()._get_policy_info(observations)
        
        # 添加3D体素地图可视化
        if self._voxel_map is not None:
            policy_info["voxel_map_3d"] = self._visualize_voxel_map()
            policy_info["voxel_statistics"] = self._voxel_map.get_statistics()
        
        return policy_info


class VoxelITMPolicy(VoxelNavPolicy):
    """结合3D体素地图和ITM的导航策略"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化ITM客户端
        self._itm = BLIP2ITMClient(port=os.environ.get("BLIP2ITM_PORT", 12182))
        
        # Value map for semantic information
        self._value_map = ValueMap(
            size=self._map_size_in_pixels,
            pixels_per_meter=self._pixels_per_meter,
            value_channels=len(self._text_prompt.split(PROMPT_SEPARATOR)),
        )
    
    def reset(self) -> None:
        super().reset()
        self._value_map.reset()
    
    def _update_value_map_3d(self) -> None:
        """更新3D语义价值地图"""
        if not hasattr(self, '_observations_cache') or not self._observations_cache.get("value_map_rgbd"):
            return
            
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]
        
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            # 更新2D value map
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)
            
            # 同时更新3D体素地图（可选：添加语义信息）
            self._update_voxel_semantic_info(rgb, depth, tf, min_depth, max_depth, cosine)
    
    def _update_voxel_semantic_info(self, rgb, depth, tf, min_depth, max_depth, semantic_scores):
        """为体素地图添加语义信息"""
        # 这里可以为体素地图添加语义标签
        # 例如：为每个体素存储其语义相似度分数
        pass
    
    def _sort_frontiers_by_value_3d(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """基于3D语义价值对前沿点排序"""
        if len(frontiers) == 0:
            return frontiers, []
        
        # 获取2D投影上的价值
        frontiers_2d = frontiers[:, :2] if frontiers.shape[1] >= 2 else frontiers
        frontiers_2d, values = super()._sort_frontiers_by_value(observations, frontiers_2d)
        
        # 如果是3D前沿点，保持高度信息
        if frontiers.shape[1] == 3:
            frontiers_3d = np.column_stack([
                frontiers_2d,
                frontiers[:len(frontiers_2d), 2]  # 保持原有高度
            ])
            return frontiers_3d, values
        
        return frontiers_2d, values
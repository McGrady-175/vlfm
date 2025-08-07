# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap  
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.radio import RADIOClient
from vlfm.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class BaseRADIOPolicy(BaseObjectNavPolicy):
    """基于RADIO模型的基础策略类"""
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        radio_port: int = 12185,  # RADIO模型的端口
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # 使用RADIO客户端替代BLIP2 ITM
        self._radio = RADIOClient(port=radio_port)
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value (RADIO): {best_value*100:.2f}%"
        print(f"Best value (RADIO): {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """使用RADIO模型选择最佳前沿点
        
        Args:
            observations: 环境观测
            frontiers: 可选择的前沿点
            
        Returns:
            最佳前沿点和其价值分数
        """
        # 排序前沿点并获取价值
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        
        # 如果存在上一个追求的点，考虑是否继续追求
        if hasattr(self, '_last_frontier') and not np.array_equal(self._last_frontier, np.zeros(2)):
            # 检查last frontier是否仍在候选列表中
            last_frontier_still_valid = False
            for i, frontier in enumerate(sorted_pts):
                if np.linalg.norm(frontier - self._last_frontier) < 0.5:  # 0.5米阈值
                    current_value = sorted_values[i]
                    # 如果当前价值与上次相比没有显著下降，继续追求
                    if current_value >= self._last_value * 0.8:  # 允许20%的价值下降
                        best_frontier_idx = i
                        last_frontier_still_valid = True
                        break
            
            if not last_frontier_still_valid:
                print("Switching to new frontier (RADIO)")

        # 如果没有有效的上一个前沿点，选择价值最高的
        if best_frontier_idx is None:
            best_frontier_idx = 0

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        
        # 更新记录
        self._last_frontier = best_frontier.copy()
        self._last_value = best_value

        return best_frontier, best_value

    def _update_value_map(self) -> None:
        """使用RADIO模型更新价值地图"""
        # 获取所有RGB观测数据
        all_rgb = [x[0] for x in self._observations_cache["value_map_rgbd"]]
        
        # 为每个提示计算RADIO相似度分数
        text_prompts = self._text_prompt.split(PROMPT_SEPARATOR)
        
        cosines = []
        for rgb in all_rgb:
            rgb_cosines = []
            for prompt in text_prompts:
                # 替换目标对象占位符
                processed_prompt = prompt.replace("target_object", self._target_object)
                # 使用RADIO计算相似度
                cosine_score = self._radio.cosine(rgb, processed_prompt)
                rgb_cosines.append(cosine_score)
            cosines.append(rgb_cosines)
        
        # 更新价值地图
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class RADIOPolicy(BaseRADIOPolicy):
    """使用RADIO模型和前沿地图的策略"""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)
        return self._frontier_map.sort_waypoints()


class RADIOPolicyV2(BaseRADIOPolicy):
    """使用RADIO模型和价值地图的改进策略"""
    
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class RADIOPolicyV3(RADIOPolicyV2):
    """带探索阈值的RADIO策略"""
    
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # 获取第一个通道的值
            first_channel = arr[:, :, 0]
            # 获取所有通道的最大值
            max_values = np.max(arr, axis=2)
            # 创建布尔掩码，第一个通道高于阈值的区域
            above_threshold = first_channel > self._exploration_thresh
            # 应用掩码
            result = np.where(above_threshold, max_values, first_channel)
            return result

        self._value_map._vis_reduce_fn = visualize_value_map
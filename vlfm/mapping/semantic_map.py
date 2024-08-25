from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_point_cloud, transform_points
from vlfm.utils.img_utils import fill_small_holes


class SemanticMap(BaseMap):
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
    # _camera_positions: List[np.ndarray] = []
    # _last_camera_yaw: float = 0.0
    # _min_confidence: float = 0.25
    # _decision_threshold: float = 0.35
    _map: np.ndarray
    _map_dtype: np.dtype = np.dtype("uint8")
    _frontiers_px: np.ndarray = np.array([])
    frontiers: np.ndarray = np.array([])
    radius_padding_color: tuple = (100, 100, 100)

    def __init__(
        self,
        min_height: float,
        max_height: float,
        agent_radius: float,
        area_thresh: float = 3.0,  # square meters
        hole_area_thresh: int = 100000,  # square pixels
        size: int = 1000,
        pixels_per_meter: int = 20,
        semantic_id: dict = None,
        multi_channel: int = 0,
    ):
        super().__init__(size, pixels_per_meter)
        self.multi_channel = multi_channel
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=np.dtype("uint8"))
        if self.multi_channel > 2:
            self._map = np.zeros((*self._map.shape, self.multi_channel), dtype=np.dtype("uint8"))
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._min_height = min_height
        self._max_height = max_height
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.semantic_id = semantic_id

    def reset(self) -> None:
        super().reset()
        self._map.fill(0)
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def update_map(
        self,
        depth: Union[np.ndarray, Any],
        semantic: Union[np.ndarray, Any],
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
        topdown_fov: float,
        explore: bool = True,
        update_semantics: bool = True,
    ) -> None:
        """
        Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.

        Args:
            depth (np.ndarray): The depth image to use for updating the semantic map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            semantic (np.ndarray): The mask to use for updating the semantic map. It has the
                same shape as the depth image, and each value of pixel refers a class id.
                ID offest 1 comparing to the original ade20k dataset.
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.
            topdown_fov (float): The field of view of the depth camera projected onto
                the topdown map.
            explore (bool): Whether to update the explored area.
            update_obstacles (bool): Whether to update the obstacle map.
        """
        if self.multi_channel:
            # multi-channel semantic map, each channel represents a class
            if update_semantics:
                if self._hole_area_thresh == -1:
                    filled_depth = np.where(depth == 0, 1.0, depth)
                else:
                    filled_depth = fill_small_holes(depth, self._hole_area_thresh)

                scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
                mask = scaled_depth < max_depth
                semantic_pcd_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy, semantic_mask=semantic)
                semantic_pcd_episodic_frame = transform_points(tf_camera_to_episodic, semantic_pcd_camera_frame)

                # 移除指定高度的点 可能存在类别信息丢失
                obstacle_cloud = filter_points_by_height(
                    semantic_pcd_episodic_frame, self._min_height, self._max_height
                )
                semantic_cloud = filter_points_by_class(obstacle_cloud, cls_id=[3, 5])

                # Populate topdown map with obstacle locations
                xy_points = semantic_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points)
                cls_ind = semantic_cloud[:, 3].astype(np.uint8)

                # fix offset for cls id
                self._map[pixel_points[:, 1], pixel_points[:, 0], cls_ind - 1] = 1

                occupied_area = np.any(self._map[:, :, :] != 0, axis=2).astype(np.uint8)

                # Update the navigable area, which is an inverse of the obstacle map after a
                # dilation operation to accommodate the robot's radius.
                self._navigable_map = 1 - cv2.dilate(
                    occupied_area.astype(np.uint8),
                    self._navigable_kernel,
                    iterations=1,
                ).astype(bool)
        else:
            # single-channel semantic map, all classes are merged into one channel, randomly
            if update_semantics:
                if self._hole_area_thresh == -1:
                    filled_depth = depth.copy()
                    filled_depth[depth == 0] = 1.0
                else:
                    filled_depth = fill_small_holes(depth, self._hole_area_thresh)
                scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
                mask = scaled_depth < max_depth
                semantic_pcd_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy, semantic_mask=semantic)
                semantic_pcd_episodic_frame = transform_points(tf_camera_to_episodic, semantic_pcd_camera_frame)
                # 移除指定高度的点 可能存在类别信息丢失
                obstacle_cloud = filter_points_by_height(
                    semantic_pcd_episodic_frame, self._min_height, self._max_height
                )
                semantic_cloud = filter_points_by_class(obstacle_cloud, cls_id=[4, 6])
                # Populate topdown map with obstacle locations
                xy_points = semantic_cloud[:, :2]
                pixel_points = self._xy_to_px(xy_points)
                self._map[pixel_points[:, 1], pixel_points[:, 0]] = semantic_cloud[:, 3]

                occupied_area = np.zeros_like(self._map, dtype=np.uint8)
                occupied_area[self._map >= 1] = 1
                # Update the navigable area, which is an inverse of the obstacle map after a
                # dilation operation to accommodate the robot's radius.
                self._navigable_map = 1 - cv2.dilate(
                    occupied_area.astype(np.uint8),
                    self._navigable_kernel,
                    iterations=1,
                ).astype(bool)

        if not explore:
            return

        # Update the explored area
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros((self._map.shape[0], self._map.shape[1]), dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        # Compute frontier locations
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """Returns the frontiers of the map."""
        # Dilate the explored area slightly to prevent small gaps between the explored
        # area and the unnavigable area from being detected as frontiers.
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def visualize(self) -> np.ndarray:
        """Visualizes the map. Using different colors for different classes."""

        mask_palette = np.array([(255, 255, 255)] + get_palette(len(self.semantic_id)))

        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # Draw explored area in light green
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # +1, avoid origin value confict with wall
        combined_map = np.argmax(self._map + 1, axis=2)
        # 将 combined_map 转换为 one-hot 编码矩阵
        one_hot_map = np.eye(len(mask_palette))[combined_map]
        # 使用矩阵乘法将 one-hot 编码矩阵与 mask_palette 相乘
        vis_img[:, :, 0] = np.dot(one_hot_map, mask_palette[:, 0])
        vis_img[:, :, 1] = np.dot(one_hot_map, mask_palette[:, 1])
        vis_img[:, :, 2] = np.dot(one_hot_map, mask_palette[:, 2])
        # Draw frontiers in blue (200, 0, 0)
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img


def filter_points_by_class(points: np.ndarray, cls_id: List[int]):
    """
    remove specific class points, filter out the points with class id in cls_id
    """
    return points[~np.isin(points[:, 3], cls_id)]


def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]


def get_palette(num_classes: int) -> np.ndarray:
    """Returns a color palette for visualizing semantic segmentation maps."""

    state = np.random.get_state()
    # random color
    np.random.seed(42)
    palette = np.random.randint(0, 256, size=(num_classes, 3))
    np.random.set_state(state)
    dataset_palette = [tuple(c) for c in palette]
    return dataset_palette

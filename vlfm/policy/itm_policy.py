# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Tuple, Union

import functools
import cv2
import numpy as np
from torch import Tensor
import torch
import time
import logging

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.semantic_map import SemanticMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
from vlfm.vlm.ssa import SSAClient
from vlfm.vlm.yolo_world import YOLOWorldClient
from vlfm.vlm.yolov10 import YOLOv10Client

from vlfm.utils.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from sentence_transformers import SentenceTransformer

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"

HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    # keep same with the aggregation function in ValueMap
    _circle_marker_radius: int = 15
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
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        log_level = os.getenv("LOG_LEVEL", "WARNING")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._text_prompt = text_prompt
        self.use_max_confidence = use_max_confidence
        self.sync_explored_areas = sync_explored_areas
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
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}% from {sorted_values}"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections, reduce_fn=None) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn if not reduce_fn else reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self) -> None:
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
            self._value_map.update_map(np.array(cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
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
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
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


class ITMPolicyV2_YOLOWORLD(BaseITMPolicy):
    """
    remove _coco_object_detector, only using yolo world for both original coco obj detection and open vocab obj detection
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # disable grounding dino, using yolo world for both original coco obj detection and open vocab obj detection
        self._object_detector = YOLOWorldClient(port=int(os.environ.get("YOLOWORLD_PORT", "12186")))
        # remove the original yolov7
        self._coco_object_detector = None

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        detections = self._object_detector.predict(img, target_classes)
        det_conf_threshold = 0.5
        detections.filter_by_conf(det_conf_threshold)

        return detections

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


class ITMPolicyV2_YOLOv10(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # using yolov10 to replace original yolov7
        self._coco_object_detector = YOLOv10Client(port=int(os.environ.get("YOLOV10_PORT", "12187")))

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


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]


#  v2 + re_initialize
class ITMPolicyV4(BaseITMPolicy):

    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.initialize_cnt = 0
        self.skip_reinitialize = False

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

        self._pre_step(observations, masks)
        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]

        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)
        print("Detected objects:", [d.num_detections for d in detections])
        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self.initialize()
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            # 在做 explore 之前 需要 check 一下很近的区域是否有其他 frontiers
            if not self.skip_reinitialize and self.check_nearby_frontiers(1.5) == True:
                self._done_initializing = False
                pointnav_action = self.initialize()
            else:
                pointnav_action = self._explore(observations)
                self.skip_reinitialize = False
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        print("Initialize: --------------------", self.initialize_cnt)

        self._done_initializing = not self.initialize_cnt < 12  # type: ignore
        self.initialize_cnt = 0 if self._done_initializing else self.initialize_cnt + 1
        self.skip_reinitialize = True if self._done_initializing else False
        return TorchActionIDs.TURN_LEFT

    def check_nearby_frontiers(self, threshold=1.0):
        robot_xy = self._observations_cache["robot_xy"]
        frontiers = self._observations_cache["frontier_sensor"]
        if len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return False
        nearest_frontier = np.linalg.norm(robot_xy - frontiers[0])
        close_frontier_cnt = 0
        for frontier in frontiers:
            distance = np.linalg.norm(robot_xy - frontier)
            if distance < nearest_frontier:
                nearest_frontier = distance
            if distance < threshold:
                close_frontier_cnt += 1
        print(f"Nearest frontier distance: {nearest_frontier:.2f}, Close frontiers count: {close_frontier_cnt}")
        if close_frontier_cnt > 1:
            return True
        else:
            return False

    # from parent BaseITMPolicy
    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}% on {best_frontier}"
        print(f"Best value: {best_value*100:.2f}% on {best_frontier}")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


# v2 + seg objs, calculate embedding similarity
class ITMPolicyV5(ITMPolicyV2):

    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ssa = SSAClient(port=12185)
        self._word2vec = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.id2label = CONFIG_ADE20K_ID2LABEL["id2label"]

    def _update_value_map(self) -> None:

        def cosine_similarity(embedding1, embedding2):
            # 计算两个向量的点积
            dot_product = np.dot(embedding1, embedding2)
            # 计算两个向量的模长
            norm_embedding1 = np.linalg.norm(embedding1)
            norm_embedding2 = np.linalg.norm(embedding2)
            # 计算余弦相似度
            similarity = dot_product / (norm_embedding1 * norm_embedding2)
            return similarity

        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        similarity_per_img = []
        for rgb in all_rgb:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = self._ssa.segment(bgr)
            unique_labels = np.unique(mask) - 1

            labels = [self.id2label[str(label_id)] for label_id in unique_labels]
            label_embeddings = self._word2vec.encode(labels, show_progress_bar=False)
            target_embedding = self._word2vec.encode(self._target_object, show_progress_bar=False)
            cosine_similarities = [
                cosine_similarity(target_embedding, label_embedding) for label_embedding in label_embeddings
            ]
            similarity_per_img.append([np.mean(cosine_similarities)])
            print(f"Current view, get items: {labels}")
        for similirity, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            similarity_per_img, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(np.array(similirity), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )


# v2 + seg frames generating semantic occupancy map
class ITMPolicyV6(BaseITMPolicy):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ssa = SSAClient(port=int(os.environ.get("SSA_PORT", "12185")))
        self._word2vec = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.id2label = CONFIG_ADE20K_ID2LABEL["id2label"]

        # 150 semantic classes, 6 target classes
        self.similarity_mat = np.zeros((len(self.id2label), len(HM3D_ID_TO_NAME)))
        for i in range(150):
            for j in range(6):
                self.similarity_mat[i][j] = self.cosine_similarity(self.id2label[str(i)], HM3D_ID_TO_NAME[j])
        print(
            f"Similarity matrix generated, contians {self.similarity_mat.shape[0]} semantic classes and {self.similarity_mat.shape[1]} target classes"
        )

        self._semantic_map = SemanticMap(min_height=0.15, max_height=0.88, agent_radius=0.18, semantic_id=self.id2label)

    def cosine_similarity(self, label, target):

        label_embedding = self._word2vec.encode(label, show_progress_bar=False)
        target_embedding = self._word2vec.encode(target, show_progress_bar=False)
        # 计算两个向量的点积
        dot_product = np.dot(label_embedding, target_embedding)
        # 计算两个向量的模长
        norm_embedding1 = np.linalg.norm(label_embedding)
        norm_embedding2 = np.linalg.norm(target_embedding)
        # 计算余弦相似度
        similarity = dot_product / (norm_embedding1 * norm_embedding2)
        return similarity

    def _update_semantic_map(self) -> None:
        # all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        for rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache["value_map_rgbd"]:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = self._ssa.segment(bgr)
            unique_labels = np.unique(mask) - 1
            labels = [self.id2label[str(label_id)] for label_id in unique_labels]
            print(f"Current view, get items: {labels}")

            self._semantic_map.update_map(
                depth, mask, tf, self._min_depth, self._max_depth, self._fx, self._fy, self._camera_fov
            )

    def _update_value_map(self) -> None:
        target_id = HM3D_ID_TO_NAME.index(self._target_object)
        self._value_map.generate_from_semantic_map(self._semantic_map._map, self.similarity_mat, target_id)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_semantic_map()
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values

    def _reset(self) -> None:
        self._semantic_map.reset()
        super()._reset()


# v7 adjust occupancy/value map generation
class ITMPolicyV7(BaseITMPolicy):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ssa = SSAClient(port=int(os.environ.get("SSA_PORT", "12185")))
        self._word2vec = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.id2label = CONFIG_ADE20K_ID2LABEL["id2label"]

        # 150 semantic classes, 6 target classes
        self.similarity_mat = np.zeros((len(self.id2label), len(HM3D_ID_TO_NAME)))
        for i in range(len(self.id2label)):
            for j in range(len(HM3D_ID_TO_NAME)):
                self.similarity_mat[i][j] = self.cosine_similarity(self.id2label[str(i)], HM3D_ID_TO_NAME[j])
        print(
            f"Similarity matrix generated, contians {self.similarity_mat.shape[0]} semantic classes and {self.similarity_mat.shape[1]} target classes"
        )

        self._semantic_map = SemanticMap(
            min_height=0.15,
            max_height=0.88,
            agent_radius=0.18,
            semantic_id=self.id2label,
            multi_channel=len(self.id2label),
        )

        self._value_map: ValueMap = ValueMap(
            value_channels=len(self.id2label),
            use_max_confidence=self.use_max_confidence,
            obstacle_map=self._obstacle_map if self.sync_explored_areas else None,
        )

        def optimized_reduce_fn(arr, exclude_value=0):
            """
            计算每个元素沿最后一个轴的非零元素和指定数值外的平均值。

            参数:
            arr (np.ndarray): 输入数组。
            exclude_value (float): 要排除的指定数值，默认为0。

            返回:
            result (np.ndarray): 沿最后一个轴的非零元素和指定数值外的平均值。
            """
            # 创建一个掩码，标记非零元素和指定数值外的元素
            mask = arr != exclude_value
            # 计算非零元素和指定数值外的数量
            count_valid = np.sum(mask, axis=-1)
            # 计算非零元素和指定数值外的总和
            sum_valid = np.sum(arr * mask, axis=-1)
            # 计算平均值，避免除以零
            result = np.divide(sum_valid, count_valid, where=count_valid != 0)
            # 将没有有效元素的位置设置为 0
            result[count_valid == 0] = 0
            return result

        self.reduce_fn_vis = optimized_reduce_fn
        # should pass exclude_value=-1 when called
        self.reduce_fn = functools.partial(optimized_reduce_fn, exclude_value=-1)

    def cosine_similarity(self, label, target):

        label_embedding = self._word2vec.encode(label, show_progress_bar=False)
        target_embedding = self._word2vec.encode(target, show_progress_bar=False)
        # 计算两个向量的点积
        dot_product = np.dot(label_embedding, target_embedding)
        # 计算两个向量的模长
        norm_embedding1 = np.linalg.norm(label_embedding)
        norm_embedding2 = np.linalg.norm(target_embedding)
        # 计算余弦相似度
        similarity = dot_product / (norm_embedding1 * norm_embedding2)
        return similarity

    def _update_semantic_map(self) -> None:
        # all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        for rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache["value_map_rgbd"]:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = self._ssa.segment(bgr)
            unique_labels = np.unique(mask) - 1
            labels = [self.id2label[str(label_id)] for label_id in unique_labels]
            print(f"Current view, get items: {labels}")

            self._semantic_map.update_map(
                depth, mask, tf, self._min_depth, self._max_depth, self._fx, self._fy, self._camera_fov
            )

    def _update_value_map(self) -> None:
        target_id = HM3D_ID_TO_NAME.index(self._target_object)
        self._value_map.generate_weight_from_semantic_map(self._semantic_map._map, self.similarity_mat, target_id)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_semantic_map()

        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, self.reduce_fn, "max")

        return sorted_frontiers, sorted_values

    def _reset(self) -> None:
        self._semantic_map.reset()
        super()._reset()

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections, self.reduce_fn_vis)
        policy_info["semantic_map"] = cv2.cvtColor(
            self._semantic_map.visualize(),
            cv2.COLOR_BGR2RGB,
        )
        return policy_info


# v8 with different reduce_fn
class ITMPolicyV8(ITMPolicyV7):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reduce_fn = lambda i: np.max(i, axis=-1)
        self.reduce_fn_vis = lambda i: np.max(i, axis=-1)


class ITMPolicyV9(ITMPolicyV7):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        target_idx = next((key for key, value in self.id2label.items() if value == self._target_object), None)

        if self._object_map.has_object(self._target_object):
            print(f"Found target object in 【ObjectMap】: {self._target_object}")
            return self._object_map.get_best_object(self._target_object, position), "ObjectMap"
        elif target_idx:
            print(f"【Target】 {self._target_object}[{int(target_idx)}]")
            if self._semantic_map.has_object(int(target_idx), pixel_thresh=400):
                print(f"Found target object in 【SemanticMap】: {self._target_object}")
                return self._semantic_map.get_best_object(int(target_idx), position), "SemanticMap"
            else:
                return None, None
        else:
            return None, None

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_semantic_map()
        self._update_value_map()

        # parent act
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        goal, source = self._get_target_object_location(robot_xy)

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            pointnav_action = self._explore(observations)
        else:
            if source == "SemanticMap":
                mode = "explore"
                stop_flag = False
            elif source == "ObjectMap":
                mode = "navigate"
                stop_flag = True
            else:
                Exception("Goal Error")
            pointnav_action = self._pointnav(goal[:2], stop=stop_flag)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")

        self._policy_info.update(self._get_policy_info(detections[0]))

        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        WALL_CEILING_FLOOR = [0, 3, 5]
        target_sim_thresh = self.similarity_mat[WALL_CEILING_FLOOR, HM3D_ID_TO_NAME.index(self._target_object)].mean()
        # 1. get value for frontiers from semantic value map
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.75, self.reduce_fn, "max")
        # 2. weighted with distance and density
        if np.any(np.array(sorted_values) > 0.5):
            # find obj with high correlated
            return sorted_frontiers, sorted_values
        elif np.all(np.array(sorted_values) < target_sim_thresh):
            print("Arounding frontiers are not with high correlated to the target, searching by the density")

            robot_xy = self._observations_cache["robot_xy"]
            # calculate normalize distances
            distances = np.linalg.norm(sorted_frontiers - robot_xy, axis=1)
            normalized_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            normalized_distances = np.nan_to_num(normalized_distances)
            normalized_densities = self.get_densities(sorted_frontiers, 2)
            weights = np.array([0.1, 0.6, 0.3])
            combined_values = np.column_stack((sorted_values, normalized_distances, normalized_densities))
            sorted_values = np.dot(combined_values, weights)

            sorted_indices = np.argsort(sorted_values)[::-1]
            sorted_frontiers = sorted_frontiers[sorted_indices]
            sorted_values = sorted_values[sorted_indices]
            return sorted_frontiers, sorted_values
        else:
            print("Arounding frontiers have some similar objects, searching by the similarity")
            robot_xy = self._observations_cache["robot_xy"]
            # calculate normalize distances
            distances = np.linalg.norm(sorted_frontiers - robot_xy, axis=1)
            normalized_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            normalized_distances = np.nan_to_num(normalized_distances)
            normalized_densities = self.get_densities(sorted_frontiers, 2)
            weights = np.array([0.5, 0.3, 0.2])
            combined_values = np.column_stack((sorted_values, normalized_distances, normalized_densities))
            sorted_values = np.dot(combined_values, weights)

            sorted_indices = np.argsort(sorted_values)[::-1]
            sorted_frontiers = sorted_frontiers[sorted_indices]
            sorted_values = sorted_values[sorted_indices]
            return sorted_frontiers, sorted_values

    def get_densities(self, frontiers, radius):
        """
        计算每个 frontier 在指定半径内的密度。

        参数:
        frontiers (np.ndarray): 形状为 (N, 2) 的数组，表示 N 个 frontiers 的坐标。
        radius (float): 指定的半径。

        返回:
        densities (np.ndarray): 形状为 (N,) 的数组，表示每个 frontier 的密度。
        """
        num_frontiers = frontiers.shape[0]
        densities = np.zeros(num_frontiers, dtype=int)

        # 计算距离矩阵
        distance_matrix = np.linalg.norm(frontiers[:, np.newaxis, :] - frontiers[np.newaxis, :, :], axis=2)

        # 计算密度
        for i in range(num_frontiers):
            densities[i] = np.sum(distance_matrix[i] <= radius) - 1  # 减去自身

        # 归一化密度
        max_density = np.max(densities)
        if max_density > 0:
            normalized_densities = (densities - np.min(densities)) / (max_density - np.min(densities))
        else:
            normalized_densities = densities
        normalized_densities = np.nan_to_num(normalized_densities)
        return normalized_densities


class ITMPolicyV9_YOLOv10(ITMPolicyV9):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # using yolov10 to replace original yolov7
        self._coco_object_detector = YOLOv10Client(port=int(os.environ.get("YOLOV10_PORT", "12187")))


class ITMPolicyV9Fusion(BaseITMPolicy):

    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ssa = SSAClient(port=int(os.environ.get("SSA_PORT", "12185")))
        self._word2vec = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.id2label = CONFIG_ADE20K_ID2LABEL["id2label"]

        # 150 semantic classes, 6 target classes
        self.similarity_mat = np.zeros((len(self.id2label), len(HM3D_ID_TO_NAME)))
        for i in range(len(self.id2label)):
            for j in range(len(HM3D_ID_TO_NAME)):
                self.similarity_mat[i][j] = self.cosine_similarity(self.id2label[str(i)], HM3D_ID_TO_NAME[j])
        print(
            f"Similarity matrix generated, contians {self.similarity_mat.shape[0]} semantic classes and {self.similarity_mat.shape[1]} target classes"
        )
        self.empty_scene_blip2_thresh = float(
            np.mean(
                [
                    self._itm.cosine(
                        np.ones((480, 640, 3)) * 255,
                        p.replace("target_object", self._target_object.replace("|", "/")),
                    )
                    for p in self._text_prompt.split(PROMPT_SEPARATOR)
                ]
            )
        )

        self._semantic_map = SemanticMap(
            min_height=0.15,
            max_height=0.88,
            agent_radius=0.18,
            semantic_id=self.id2label,
            multi_channel=len(self.id2label),
        )

        self._semantic_value_map: ValueMap = ValueMap(
            value_channels=len(self.id2label),
            use_max_confidence=self.use_max_confidence,
            obstacle_map=self._obstacle_map if self.sync_explored_areas else None,
        )

        def optimized_reduce_fn(arr, exclude_value=0):
            """
            计算每个元素沿最后一个轴的非零元素和指定数值外的平均值。

            参数:
            arr (np.ndarray): 输入数组。
            exclude_value (float): 要排除的指定数值，默认为0。

            返回:
            result (np.ndarray): 沿最后一个轴的非零元素和指定数值外的平均值。
            """
            # 创建一个掩码，标记非零元素和指定数值外的元素
            mask = arr != exclude_value
            # 计算非零元素和指定数值外的数量
            count_valid = np.sum(mask, axis=-1)
            # 计算非零元素和指定数值外的总和
            sum_valid = np.sum(arr * mask, axis=-1)
            # 计算平均值，避免除以零
            result = np.divide(sum_valid, count_valid, where=count_valid != 0)
            # 将没有有效元素的位置设置为 0
            result[count_valid == 0] = 0
            return result

        self.reduce_fn_vis = optimized_reduce_fn
        # should pass exclude_value=-1 when called
        self.reduce_fn = functools.partial(optimized_reduce_fn, exclude_value=-1)

    def cosine_similarity(self, label, target):

        label_embedding = self._word2vec.encode(label, show_progress_bar=False)
        target_embedding = self._word2vec.encode(target, show_progress_bar=False)
        # 计算两个向量的点积
        dot_product = np.dot(label_embedding, target_embedding)
        # 计算两个向量的模长
        norm_embedding1 = np.linalg.norm(label_embedding)
        norm_embedding2 = np.linalg.norm(target_embedding)
        # 计算余弦相似度
        similarity = dot_product / (norm_embedding1 * norm_embedding2)
        return similarity

    def _update_semantic_map(self) -> None:
        for rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache["value_map_rgbd"]:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = self._ssa.segment(bgr)
            unique_labels = np.unique(mask) - 1
            labels = [self.id2label[str(label_id)] for label_id in unique_labels]
            print(f"Current view, get items: {labels}")

            self._semantic_map.update_map(
                depth, mask, tf, self._min_depth, self._max_depth, self._fx, self._fy, self._camera_fov
            )

    def _update_semantic_value_map(self) -> None:
        target_id = HM3D_ID_TO_NAME.index(self._target_object)
        self._semantic_value_map.generate_weight_from_semantic_map(
            self._semantic_map._map, self.similarity_mat, target_id
        )

        self._semantic_value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_semantic_map()
        self._update_value_map()
        self._update_semantic_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        WALL_CEILING_FLOOR = [0, 3, 5]
        target_sim_thresh = np.mean(
            [
                self.empty_scene_blip2_thresh,
                self.similarity_mat[WALL_CEILING_FLOOR, HM3D_ID_TO_NAME.index(self._target_object)].mean(),
            ]
        )
        # 1. get value for frontiers from semantic value map
        _, frointer_values_blip2 = self._value_map.sort_waypoints(frontiers, 0.5, sorted=False)
        _, frontier_values_semantic = self._semantic_value_map.sort_waypoints(
            frontiers, 0.75, self.reduce_fn, "max", sorted=False
        )
        values = np.mean([frontier_values_semantic, frointer_values_blip2], axis=0)
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([frontiers[i] for i in sorted_inds])

        # 2. weighted with distance and density
        if np.all(np.array(sorted_values) < target_sim_thresh):
            print("Arounding frontiers are not with high correlated to the target, searching by the density")

            robot_xy = self._observations_cache["robot_xy"]
            # calculate normalize distances
            distances = np.linalg.norm(sorted_frontiers - robot_xy, axis=1)
            normalized_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            normalized_distances = np.nan_to_num(normalized_distances)
            normalized_densities = self.get_densities(sorted_frontiers, 2)
            weights = np.array([0.2, 0.4, 0.4])
            combined_values = np.column_stack((sorted_values, normalized_distances, normalized_densities))
            sorted_values = np.dot(combined_values, weights)

            sorted_indices = np.argsort(sorted_values)[::-1]
            sorted_frontiers = sorted_frontiers[sorted_indices]
            sorted_values = sorted_values[sorted_indices]
            return sorted_frontiers, sorted_values
        else:
            print("Arounding frontiers have some similar objects, searching by the similarity")
            robot_xy = self._observations_cache["robot_xy"]
            # calculate normalize distances
            distances = np.linalg.norm(sorted_frontiers - robot_xy, axis=1)
            normalized_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            normalized_distances = np.nan_to_num(normalized_distances)
            normalized_densities = self.get_densities(sorted_frontiers, 2)
            weights = np.array([0.8, 0.1, 0.1])
            combined_values = np.column_stack((sorted_values, normalized_distances, normalized_densities))
            sorted_values = np.dot(combined_values, weights)

            sorted_indices = np.argsort(sorted_values)[::-1]
            sorted_frontiers = sorted_frontiers[sorted_indices]
            sorted_values = sorted_values[sorted_indices]
            return sorted_frontiers, sorted_values

    def get_densities(self, frontiers, radius):
        """
        计算每个 frontier 在指定半径内的密度。

        参数:
        frontiers (np.ndarray): 形状为 (N, 2) 的数组，表示 N 个 frontiers 的坐标。
        radius (float): 指定的半径。

        返回:
        densities (np.ndarray): 形状为 (N,) 的数组，表示每个 frontier 的密度。
        """
        num_frontiers = frontiers.shape[0]
        densities = np.zeros(num_frontiers, dtype=int)

        # 计算距离矩阵
        distance_matrix = np.linalg.norm(frontiers[:, np.newaxis, :] - frontiers[np.newaxis, :, :], axis=2)

        # 计算密度
        for i in range(num_frontiers):
            densities[i] = np.sum(distance_matrix[i] <= radius) - 1  # 减去自身

        # 归一化密度
        max_density = np.max(densities)
        if max_density > 0:
            normalized_densities = (densities - np.min(densities)) / (max_density - np.min(densities))
        else:
            normalized_densities = densities
        normalized_densities = np.nan_to_num(normalized_densities)
        return normalized_densities

    def _reset(self) -> None:
        self._semantic_map.reset()
        self._semantic_value_map.reset()
        super()._reset()

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections, self.reduce_fn_vis)
        policy_info["semantic_map"] = cv2.cvtColor(
            self._semantic_map.visualize(),
            cv2.COLOR_BGR2RGB,
        )
        return policy_info


class ITMPolicyV11(BaseITMPolicy):
    # ITMPolicyV9FusionWeightedSemanticMap
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ssa = SSAClient(port=int(os.environ.get("SSA_PORT", "12185")))
        self._word2vec = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.id2label = CONFIG_ADE20K_ID2LABEL["id2label"]

        # 150 semantic classes, 6 target classes
        self.similarity_mat = np.zeros((len(self.id2label), len(HM3D_ID_TO_NAME)))
        for i in range(len(self.id2label)):
            for j in range(len(HM3D_ID_TO_NAME)):
                self.similarity_mat[i][j] = self.cosine_similarity(self.id2label[str(i)], HM3D_ID_TO_NAME[j])
        self.logger.debug(
            f"Similarity matrix generated, contians {self.similarity_mat.shape[0]} semantic classes and {self.similarity_mat.shape[1]} target classes"
        )
        self.empty_scene_blip2_thresh = float(
            np.mean(
                [
                    self._itm.cosine(
                        np.ones((480, 640, 3)) * 255,
                        p.replace("target_object", self._target_object.replace("|", "/")),
                    )
                    for p in self._text_prompt.split(PROMPT_SEPARATOR)
                ]
            )
        )

        self._semantic_map = SemanticMap(
            min_height=0.15,
            max_height=0.88,
            agent_radius=0.18,
            semantic_id=self.id2label,
            multi_channel=len(self.id2label),
        )

        self._semantic_value_map: ValueMap = ValueMap(
            value_channels=len(self.id2label),
            use_max_confidence=self.use_max_confidence,
            obstacle_map=self._obstacle_map if self.sync_explored_areas else None,
        )

        def optimized_reduce_fn(arr, exclude_value=0):
            """
            计算每个元素沿最后一个轴的非零元素和指定数值外的平均值。

            参数:
            arr (np.ndarray): 输入数组。
            exclude_value (float): 要排除的指定数值，默认为0。

            返回:
            result (np.ndarray): 沿最后一个轴的非零元素和指定数值外的平均值。
            """
            # 创建一个掩码，标记非零元素和指定数值外的元素
            mask = arr != exclude_value
            # 计算非零元素和指定数值外的数量
            count_valid = np.sum(mask, axis=-1)
            # 计算非零元素和指定数值外的总和
            sum_valid = np.sum(arr * mask, axis=-1)
            # 计算平均值，避免除以零
            result = np.divide(sum_valid, count_valid, where=count_valid != 0)
            # 将没有有效元素的位置设置为 0
            result[count_valid == 0] = 0
            return result

        self.reduce_fn_vis = optimized_reduce_fn
        # should pass exclude_value=-1 when called
        self.reduce_fn = functools.partial(optimized_reduce_fn, exclude_value=-1)

    def cosine_similarity(self, label, target):

        label_embedding = self._word2vec.encode(label, show_progress_bar=False)
        target_embedding = self._word2vec.encode(target, show_progress_bar=False)
        # 计算两个向量的点积
        dot_product = np.dot(label_embedding, target_embedding)
        # 计算两个向量的模长
        norm_embedding1 = np.linalg.norm(label_embedding)
        norm_embedding2 = np.linalg.norm(target_embedding)
        # 计算余弦相似度
        similarity = dot_product / (norm_embedding1 * norm_embedding2)
        return similarity

    def _update_semantic_map(self) -> None:
        for rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache["value_map_rgbd"]:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = self._ssa.segment(bgr)
            unique_labels = np.unique(mask) - 1
            labels = [self.id2label[str(label_id)] for label_id in unique_labels]
            self.logger.debug(f"Current view, get items: {labels}")
            self._semantic_map.update_map(
                depth, mask, tf, min_depth, max_depth, self._fx, self._fy, fov, weighted="conf_weighted"
            )

    def _update_semantic_value_map(self) -> None:
        target_id = HM3D_ID_TO_NAME.index(self._target_object)
        self._semantic_value_map.generate_weight_from_semantic_map(
            self._semantic_map._map, self.similarity_mat, target_id
        )

        self._semantic_value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_semantic_map()
        self._update_value_map()
        self._update_semantic_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        WALL_CEILING_FLOOR = [0, 3, 5]
        target_sim_thresh = np.mean(
            [
                self.empty_scene_blip2_thresh,
                self.similarity_mat[WALL_CEILING_FLOOR, HM3D_ID_TO_NAME.index(self._target_object)].mean(),
            ]
        )
        # 1. get value for frontiers from semantic value map
        _, frointer_values_blip2 = self._value_map.sort_waypoints(frontiers, 0.5, sorted=False)
        _, frontier_values_semantic = self._semantic_value_map.sort_waypoints(
            frontiers, 0.75, self.reduce_fn, "max", sorted=False
        )
        values = np.mean([frontier_values_semantic, frointer_values_blip2], axis=0)
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([frontiers[i] for i in sorted_inds])
        self.logger.debug(f"Frointer values: Semantic-{frontier_values_semantic}, Blip2-{frointer_values_blip2}")
        # 2. weighted with distance and density
        if np.all(np.array(sorted_values) < target_sim_thresh):
            self.logger.debug(
                "Arounding frontiers are not with high correlated to the target, searching by the density"
            )
            robot_xy = self._observations_cache["robot_xy"]
            # calculate normalize distances
            distances = np.linalg.norm(sorted_frontiers - robot_xy, axis=1)
            normalized_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            normalized_distances = np.nan_to_num(normalized_distances)
            normalized_densities = self.get_densities(sorted_frontiers, 2)
            weights = np.array([0.2, 0.4, 0.4])
            self.logger.debug(f"Normalized distances: {normalized_distances}, densities: {normalized_densities}")
            combined_values = np.column_stack((sorted_values, normalized_distances, normalized_densities))
            sorted_values = np.dot(combined_values, weights)

            sorted_indices = np.argsort(sorted_values)[::-1]
            sorted_frontiers = sorted_frontiers[sorted_indices]
            sorted_values = sorted_values[sorted_indices]
            return sorted_frontiers, sorted_values
        else:
            self.logger.debug("Arounding frontiers have some similar objects, searching by the similarity")
            robot_xy = self._observations_cache["robot_xy"]
            # calculate normalize distances
            distances = np.linalg.norm(sorted_frontiers - robot_xy, axis=1)
            normalized_distances = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            normalized_distances = np.nan_to_num(normalized_distances)
            normalized_densities = self.get_densities(sorted_frontiers, 2)
            weights = np.array([0.8, 0.1, 0.1])
            combined_values = np.column_stack((sorted_values, normalized_distances, normalized_densities))
            sorted_values = np.dot(combined_values, weights)

            sorted_indices = np.argsort(sorted_values)[::-1]
            sorted_frontiers = sorted_frontiers[sorted_indices]
            sorted_values = sorted_values[sorted_indices]
            return sorted_frontiers, sorted_values

    def get_densities(self, frontiers, radius):
        """
        计算每个 frontier 在指定半径内的密度。

        参数:
        frontiers (np.ndarray): 形状为 (N, 2) 的数组，表示 N 个 frontiers 的坐标。
        radius (float): 指定的半径。

        返回:
        densities (np.ndarray): 形状为 (N,) 的数组，表示每个 frontier 的密度。
        """
        num_frontiers = frontiers.shape[0]
        densities = np.zeros(num_frontiers, dtype=int)

        # 计算距离矩阵
        distance_matrix = np.linalg.norm(frontiers[:, np.newaxis, :] - frontiers[np.newaxis, :, :], axis=2)

        # 计算密度
        for i in range(num_frontiers):
            densities[i] = np.sum(distance_matrix[i] <= radius) - 1  # 减去自身

        # 归一化密度
        max_density = np.max(densities)
        if max_density > 0:
            normalized_densities = (densities - np.min(densities)) / (max_density - np.min(densities))
        else:
            normalized_densities = densities
        normalized_densities = np.nan_to_num(normalized_densities)
        return normalized_densities

    def _reset(self) -> None:
        self._semantic_map.reset()
        self._semantic_value_map.reset()
        super()._reset()

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections, self.reduce_fn_vis)
        policy_info["semantic_map"] = cv2.cvtColor(
            self._semantic_map.visualize(),
            cv2.COLOR_BGR2RGB,
        )
        policy_info["semantic_value_map"] = cv2.cvtColor(
            self._semantic_value_map.visualize(),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info


class ITMPolicyV10(ITMPolicyV9):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reduce_fn = lambda i: np.max(i, axis=-1)
        self.reduce_fn_vis = lambda i: np.max(i, axis=-1)

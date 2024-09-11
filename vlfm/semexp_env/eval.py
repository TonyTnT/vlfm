import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import sys
import csv


try:
    from vlfm.object_goal_navigation.arguments import get_args
    from vlfm.object_goal_navigation.envs import make_vec_envs
except Exception as e:
    print(f"Object Goal Navigation not found. {e}")

from moviepy.editor import ImageSequenceClip

from vlfm.semexp_env.semexp_policy import (
    SemExpITMPolicyV2,
    SemExpITMPolicyV3,
    SemExpITMPolicyV11,
    SemExpITMPolicyV12,
    SemExpITMPolicyV13,
    SemExpITMPolicyV14,
    SemExpITMPolicyV15,
    SemExpITMPolicyV16,
)
from vlfm.utils.img_utils import reorient_rescale_map, resize_images
from vlfm.utils.log_saver import is_evaluated, log_episode
from vlfm.utils.visualization import add_text_to_image
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"

args = get_args()
args.num_processes = 1
args.num_processes_on_first_gpu = 1
args.agent = "vlfm"  # Doesn't really matter as long as it's not "sem_exp"
args.split = "val"
args.task_config = "/home/chenx2/vlfm/vlfm/semexp_env/objnav_gibson_vlfm.yaml"
args.scene_name = os.environ.get("SCENE_NAME", "EMPTY")
# Ensure a random seed
# args.seed = int(time.time() * 1000) % 2**32
args.seed = 42

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main() -> None:
    num_episodes = int(args.num_eval_episodes)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    policy_kwargs = dict(
        text_prompt="Seems like there is a target_object ahead.",
        # pointnav_policy_path="data/spot_pointnav_weights.pth",
        pointnav_policy_path="data/pointnav_weights.pth",
        depth_image_shape=(224, 224),
        pointnav_stop_radius=0.9,
        use_max_confidence=False,
        object_map_erosion_size=5,
        exploration_thresh=0.0,
        obstacle_map_area_threshold=1.5,  # in square meters
        min_obstacle_height=0.61,
        max_obstacle_height=0.88,
        hole_area_thresh=100000,
        use_vqa=False,
        vqa_prompt="Is this ",
        coco_threshold=0.8,
        non_coco_threshold=0.4,
        camera_height=0.88,
        min_depth=0.5,
        max_depth=5.0,
        camera_fov=79,
        image_width=640,
        visualize=True,
    )

    policy_name = os.environ.get("POLICY_NAME", None)
    policy_cls = globals()[policy_name]
    policy = policy_cls(**policy_kwargs)  # type: ignore

    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    print("Environment created", envs)
    ep_id, scene_id, target_object = "", "", ""
    sc = []
    obs, infos = envs.reset()
    spl = []
    for ep_num in tqdm(range(num_episodes)):
        vis_imgs = []
        for step in range(args.max_episode_length):
            if step == 0:
                masks = torch.zeros(1, 1, device=obs.device)
                ep_id, scene_id = infos[0]["episode_id"], infos[0]["scene_id"]
                target_object = infos[0]["goal_name"]
                print("Episode:", ep_id, "Scene:", scene_id)
            else:
                masks = torch.ones(1, 1, device=obs.device)

            if "ZSOS_LOG_DIR" in os.environ and is_evaluated(ep_id, scene_id):
                print(f"Episode {ep_id} in scene {scene_id} already evaluated")
                # Call stop action to move on to the next episode
                obs, rew, done, infos = envs.step(torch.tensor([0], dtype=torch.long))
            else:
                obs_dict = merge_obs_infos(obs, infos)
                action, policy_infos = policy.act(obs_dict, None, None, masks)

                if "VIDEO_DIR" in os.environ:
                    frame = create_frame(policy_infos)
                    frame = add_text_to_image(frame, "Step: " + str(step), top=True)
                    vis_imgs.append(frame)

                action = action.squeeze(0)

                obs, rew, done, infos = envs.step(action)

            if done:
                sc.append(infos[0]["success"])
                spl.append(infos[0]["spl"])
                print("Success:", infos[0]["success"], "--- Rate : ", np.mean(sc))
                print("SPL:", infos[0]["spl"])
                data = {
                    "success": infos[0]["success"],
                    "spl": infos[0]["spl"],
                    "distance_to_goal": infos[0]["distance_to_goal"],
                    "target_object": target_object,
                }
                folder = "../VLFMExp/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                csv_file = os.path.join(folder, f"{policy_name}_{scene_id}_metrics.csv")
                data["episode_id"] = ep_id
                data["scene_id"] = scene_id

                # Check if the CSV file exists
                file_exists = os.path.isfile(csv_file)
                # Open the CSV file in append mode
                with open(csv_file, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    # Write the header row if the file doesn't exist
                    if not file_exists:
                        writer.writeheader()
                    # Write the metrics as a new row
                    writer.writerow(data)
                if "VIDEO_DIR" in os.environ:
                    try:
                        generate_video(vis_imgs, ep_id, scene_id, data)
                    except Exception:
                        print("Error generating video")
                if "ZSOS_LOG_DIR" in os.environ and not is_evaluated(ep_id, scene_id):
                    log_episode(ep_id, scene_id, data)
                break
    print("-" * 20)
    print("Success rate:", np.mean(sc), "SPL:", np.mean(spl))
    print("-" * 20)
    print("Test successfully completed")


def merge_obs_infos(obs: torch.Tensor, infos: Tuple[Dict, ...]) -> Dict[str, torch.Tensor]:
    """Merge the observations and infos into a single dictionary."""
    rgb = obs[:, :3, ...].permute(0, 2, 3, 1)
    depth = obs[:, 3:4, ...].permute(0, 2, 3, 1)
    info_dict = infos[0]

    def tensor_from_numpy(tensor: torch.Tensor, numpy_array: np.ndarray) -> torch.Tensor:
        device = tensor.device
        new_tensor = torch.tensor(numpy_array).to(device)
        return new_tensor

    obs_dict = {
        "rgb": rgb,
        "depth": depth,
        "objectgoal": info_dict["goal_name"].replace("-", " "),
        "gps": tensor_from_numpy(obs, info_dict["gps"]).unsqueeze(0),
        "compass": tensor_from_numpy(obs, info_dict["compass"]).unsqueeze(0),
        "heading": tensor_from_numpy(obs, info_dict["heading"]).unsqueeze(0),
    }

    return obs_dict


def create_frame(policy_infos: Dict[str, Any]) -> np.ndarray:
    vis_imgs = []
    for k in ["annotated_rgb", "annotated_depth", "obstacle_map", "value_map"]:
        img = policy_infos[k]
        if "map" in k:
            img = reorient_rescale_map(img)
        if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
            # Put text in the middle saying "Target not curently detected"
            text = "Target not currently detected"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
            cv2.putText(
                img,
                text,
                (img.shape[1] // 2 - text_size[0] // 2, img.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                1,
            )
        vis_imgs.append(img)
    vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
    return vis_img


def generate_video(frames: List[np.ndarray], ep_id: str, scene_id: str, infos: Dict[str, Any]) -> None:
    """
    Saves the given list of rgb frames as a video at 10 FPS. Uses the infos to get the
    files name, which should contain the following:
        - episode_id
        - scene_id
        - success
        - spl
        - dtg
        - goal_name

    """
    video_dir = os.environ.get("VIDEO_DIR", "video_dir")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    episode_id = int(ep_id)
    success = int(infos["success"])
    spl = infos["spl"]
    dtg = infos["distance_to_goal"]
    goal_name = infos["target_object"]
    filename = (
        f"epid={episode_id:03d}-scid={scene_id}-succ={success}-spl={spl:.2f}-dtg={dtg:.2f}-target={goal_name}.mp4"
    )

    filename = os.path.join(video_dir, filename)
    # Create a video clip from the frames
    clip = ImageSequenceClip(frames, fps=10)

    # Write the video file
    clip.write_videofile(filename)


if __name__ == "__main__":
    main()

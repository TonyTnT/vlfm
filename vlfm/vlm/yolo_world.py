import sys
from typing import List, Optional

import PIL
import cv2
import numpy as np
import torch
import os
from torchvision.ops import nms

try:

    import supervision as sv
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.runner import Runner
    from mmdet.apis import init_detector
    from mmdet.utils import get_test_pipeline_cfg

except Exception:
    print("Could not import YOLOWorldServer. This is OK if you are only using the client.")

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image


class YOLOWorld:

    def __init__(self, config_file: str = None, checkpoint: str = None, score_thr=0.05, max_num_boxes=1):
        """Loads the model and saves it to a field."""
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.score_thr = score_thr
        self.max_num_boxes = max_num_boxes
        if config_file is None:
            config_file = (
                "data/yolo_world/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
            )
        if checkpoint is None:
            checkpoint = "data/yolo_world/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"

        cfg = Config.fromfile(config_file)
        print(f"Config file: {config_file}")
        # init model
        cfg.load_from = checkpoint

        self.model = init_detector(cfg, checkpoint=checkpoint, device=self.device)
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(test_pipeline_cfg)

        # Warm-up
        if self.device.type != "cpu":
            dummy_img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
            for i in range(3):
                data_info = dict(img=dummy_img, img_id=0, texts=["anything"])
                data_info = self.pipeline(data_info)
                data_batch = dict(inputs=data_info["inputs"].unsqueeze(0), data_samples=[data_info["data_samples"]])

                with torch.inference_mode():
                    _ = self.model.test_step(data_batch)[0]

    def predict(
        self,
        image: np.ndarray,
        targets: Optional[List[str]] = [["anything"]],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            targets (Optional[List[str]]): List of target classes to filter by. Defaults to [["anything"]].
            conf_thres (float): Confidence threshold for filtering detections. Defaults to 0.25.
            iou_thres (float): IOU threshold for filtering detections. Defaults to 0.45.

        Returns:
            ObjectDetections: ObjectDetections instance containing the predicted bounding boxes, scores, and classes.
        """
        orig_shape = image.shape
        # accept image format with RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_info = self.pipeline(dict(img_id=0, img=img, texts=targets))
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        with torch.inference_mode():
            output = self.model.test_step(data_batch)[0]
            pred_instances = output.pred_instances

        # nms
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=iou_thres)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > conf_thres]

        if len(pred_instances.scores) > self.max_num_boxes:
            indices = pred_instances.scores.float().topk(self.max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        # predictions
        pred_instances = pred_instances.cpu().numpy()

        # Rescale boxes from img_size to im0 size
        pred_instances.bboxes[:, 0] /= orig_shape[1]
        pred_instances.bboxes[:, 1] /= orig_shape[0]
        pred_instances.bboxes[:, 2] /= orig_shape[1]
        pred_instances.bboxes[:, 3] /= orig_shape[0]
        # pred_instances.bboxes = pred_instances.bboxes.tolist()
        detections = ObjectDetections(
            pred_instances["bboxes"], pred_instances["scores"], targets, image_source=image, fmt="xyxy"
        )

        return detections


class YOLOWorldClient:
    def __init__(self, port: int = 12186):
        self.url = f"http://localhost:{port}/yolo_world"

    def predict(self, image_numpy: np.ndarray, target_str: List) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, target=target_str)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12186)
    args = parser.parse_args()

    print("Loading model...")

    class YOLOWorldServer(ServerMixin, YOLOWorld):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            targets = [payload["target"]]
            return self.predict(image, targets).to_json()

    yolo = YOLOWorldServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolo, name="yolo_world", port=args.port)

    # # Instantiate model
    # model = YOLOWorld()
    # img_path = "data/bus.jpg"
    # img = cv2.imread(img_path)
    # # Convert to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # Predict
    # pred = model.predict(img, targets=[["bus"]])

    # print("Pred")
    # print(pred)

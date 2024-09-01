# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
from typing import List, Optional

import cv2
import numpy as np
import torch

from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

sys.path.insert(0, "yolov10/")
try:
    from ultralytics import YOLOv10 as YOLOv10u

except Exception:
    print("Could not import yolov10. This is OK if you are only using the client.")
sys.path.pop(0)


class YOLOv10:
    def __init__(self, weights: str, image_size: int = 640):
        """Loads the model and saves it to a field."""
        import ultralytics

        print(ultralytics.__version__)
        self.model = YOLOv10u.from_pretrained("jameslahm/yolov10x")
        self.image_size = image_size

        # Warm-up
        # dummy_img = torch.rand(1, 3, int(self.image_size * 0.7), self.image_size)
        # if self.half_precision:
        #     dummy_img = dummy_img.half()
        # for i in range(3):
        #     self.model(dummy_img)

    def predict(
        self,
        image: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.

        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # need to convert to RGB
        pred = self.model.predict(img)

        boxes = pred[0].boxes.xyxy.cpu().numpy()
        logits = pred[0].boxes.conf.cpu().numpy()
        phrases = [COCO_CLASSES[int(i)] for i in pred[0].boxes.cls.cpu().numpy()]

        detections = ObjectDetections(boxes, logits, phrases, image_source=image, fmt="xyxy")

        return detections


class YOLOv10Client:
    def __init__(self, port: int = 12187):
        self.url = f"http://localhost:{port}/yolov10"

    def predict(self, image_numpy: np.ndarray) -> ObjectDetections:
        print(f"YOLOv10Client")

        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12187)
    args = parser.parse_args()

    print("Loading model...")

    class YOLOv10Server(ServerMixin, YOLOv10):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    yolo = YOLOv10Server("data/yolov7-e6e.pt")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolo, name="yolov10", port=args.port)

    # # Instantiate model
    # model = YOLOv10(weights="jameslahm/yolov10x")
    # img_path = "data/bus.jpg"
    # img = cv2.imread(img_path)
    # # Convert to RGB
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # Predict
    # pred = model.predict(img)
    # print("Pred")
    # print(pred)

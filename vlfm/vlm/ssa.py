from typing import Any, List, Optional

import numpy as np
import cv2

from .server_wrapper import (
    send_request,
    str_to_image,
    str_to_uint8_arr,
)


class SSAClient:
    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/ssa"

    def segment(self, image: np.ndarray) -> np.ndarray:
        response = send_request(self.url, image=image)
        seg_mask_str = response["seg_mask"]
        h, w = image.shape[:2]
        seg_mask_str = str_to_uint8_arr(seg_mask_str, (h, w))
        # avoid id = 0 confilct
        return seg_mask_str + 1


if __name__ == "__main__":
    img = cv2.imread("temp_output/rgb_26.png", 1)
    client = SSAClient(port=12185)
    seg_mask = client.segment(img)
    # cv2.imshow("seg_mask", seg_mask)
    # cv2.waitKey(0)

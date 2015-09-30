from libpydarknet import DarknetObjectDetector, BBox

from PIL import Image
import numpy as np
import time

class Darknet_ObjectDetector():

    def __init__(self, spec, weight):
        self._detector = DarknetObjectDetector(spec, weight)

    def detect_object(self, pil_image):
        start = time.time()

        data = np.array(pil_image).transpose([2,0,1]).astype(np.uint8)

        rst = self._detector.detect_object(data.tostring(), pil_image.size[0], pil_image.size[1], 3)

        end = time.time()

        return rst, end-start

    @staticmethod
    def set_device(gpu_id):
        DarknetObjectDetector.set_device(gpu_id)

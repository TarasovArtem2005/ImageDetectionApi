from ultralytics import YOLO
import numpy as np
import cv2
from products import object_dict


class Detector:
    detection_model = YOLO("best (2).pt")

    def __init__(self, file_image: bytes):
        self.file_image = Detector.convert(file_image)

    @staticmethod
    def convert(file_image: bytes):
        np_array = np.frombuffer(file_image, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    def detect(self) -> dict:
        results = Detector.detection_model(self.file_image)
        pri = []
        for r in results:
            for c in r.boxes.cls:
                obj_index = int(Detector.detection_model.names[int(c)])
                if obj_index in object_dict:
                    pri.append(object_dict[obj_index])
        if pri:
            return {"products": pri}
        return {"products": "No product detected"}
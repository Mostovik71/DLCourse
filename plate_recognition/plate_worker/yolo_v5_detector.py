import torch
import numpy as np
from typing import List
import os
from modelhub_client import ModelHub

model_config_urls = [
    # object detection
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolov5/model-2.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolox/model-1.json",
    "http://models.vsp.net.ua/config_model/nomeroff-net-yolov5_brand_np/model-2.json",
]

# download and append to path yolo repo
local_storage = os.environ.get('LOCAL_STORAGE', os.path.join(os.path.dirname(__file__), "data"))
modelhub = ModelHub(model_config_urls=model_config_urls,
                    local_storage=local_storage)

info = modelhub.download_repo_for_model("yolov5")
repo_path = info["repo_path"]


def get_device_torch() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector(object):
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self, numberplate_classes=None) -> None:
        self.model = None
        self.numberplate_classes = ["numberplate"]
        if numberplate_classes is not None:
            self.numberplate_classes = numberplate_classes
        self.device = get_device_torch()

    def load_model(self, weights: str, device: str = '') -> None:
        device = device or self.device
        model = torch.hub.load(repo_path, 'custom', device="cpu", path=weights, source="local")
        model.to(device)
        if device != 'cpu':  # half precision only supported on CUDA
            model.half()  # to FP16

        self.model = model
        self.device = device

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name("yolov5")
            path_to_model = model_info["path"]
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
        self.load_model(path_to_model)

    @torch.no_grad()
    def predict(self, imgs: List[np.ndarray], min_accuracy: float = 0.5) -> np.ndarray:
        model_outputs = self.model(imgs)
        model_outputs = [[[item["xmin"], item["ymin"], item["xmax"], item["ymax"], item["confidence"], item["class"]]
                         for item in img_item.to_dict(orient="records")
                         if item["confidence"] > min_accuracy]
                         for img_item in model_outputs.pandas().xyxy]
        return np.array(model_outputs)

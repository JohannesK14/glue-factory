import sys
from pathlib import Path

import torchvision.transforms as transforms

from ..base_model import BaseModel

ripe_path = Path(__file__).parent / "../../../thirdparty/ripepp"

print(f"RIPE++ Path: {ripe_path.resolve()}")
# check if the path exists
if not ripe_path.exists():
    raise RuntimeError(f"RIPE++ path not found: {ripe_path}")

sys.path.append(str(ripe_path))

from ripepp import load_model_from_checkpoint, resolve_variant_checkpoint


class Ripe(BaseModel):
    default_conf = {
        "name": "RIPE++",
        "dense_outputs": False,
        "inference_conf": None,
        "variant": "default"
    }

    required_data_keys = ["image"]

    # Initialize the line matcher
    def _init(self, conf):
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ckpt_path = resolve_variant_checkpoint(conf["variant"])
        self.model = load_model_from_checkpoint(Path(ckpt_path))

        self.set_initialized()

    def _forward(self, data):
        image = data["image"]

        keypoints, scores, descriptors = [], [], []

        if self.conf.dense_outputs:
            raise NotImplementedError("Dense outputs are not supported")
        else:
            # check if image is RGB
            if image.shape[1] == 3:
                image = self.normalizer(image)

            inf_conf = self.conf.inference_conf if self.conf.inference_conf is not None else {}

            keypoints, descriptors, scores = self.model.detectAndCompute(image, **inf_conf)

        pred = {
            # "keypoints": keypoints.to(image) + 0.5,
            "keypoints": keypoints.to(image),
            "keypoint_scores": scores.to(image),
            "descriptors": descriptors.to(image),
        }

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

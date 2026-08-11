import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms

from ..base_model import BaseModel

ripe_path = Path(__file__).parent / "../../../thirdparty/ripe"

print(f"RIPE Path: {ripe_path.resolve()}")
# check if the path exists
if not ripe_path.exists():
    raise RuntimeError(f"RIPE path not found: {ripe_path}")

sys.path.append(str(ripe_path))

from ripe import vgg_hyper


class RIPE(BaseModel):
    default_conf = {
        "name": "RIPE",
        "model_path": None,
        "chunk": 4,
        "dense_outputs": False,
        "threshold": 1.0,
        "top_k": 2048,
    }

    required_data_keys = ["image"]

    # Initialize the line matcher
    def _init(self, conf):
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = vgg_hyper(model_path=conf.model_path)
        self.model.eval()

        self.set_initialized()

    def _forward(self, data):
        image = data["image"]

        keypoints, scores, descriptors = [], [], []

        chunk = self.conf.chunk

        for i in range(0, image.shape[0], chunk):
            if self.conf.dense_outputs:
                raise NotImplementedError("Dense outputs are not supported")
            else:
                im = image[: min(image.shape[0], i + chunk)]
                im = self.normalizer(im)

                H, W = im.shape[-2:]

                kpt, desc, score = self.model.detectAndCompute(
                    im,
                    threshold=self.conf.threshold,
                    top_k=self.conf.top_k,
                )
            keypoints += [kpt.squeeze(0)]
            scores += [score.squeeze(0)]
            descriptors += [desc.squeeze(0)]

            del kpt
            del desc
            del score

        keypoints = torch.stack(keypoints, 0)
        scores = torch.stack(scores, 0)
        descriptors = torch.stack(descriptors, 0)

        pred = {
            # "keypoints": keypoints.to(image) + 0.5,
            "keypoints": keypoints.to(image),
            "keypoint_scores": scores.to(image),
            "descriptors": descriptors.to(image),
        }

        return pred

    def loss(self, pred, data):
        raise NotImplementedError
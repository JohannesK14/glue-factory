from omegaconf import OmegaConf

from ...utils.checkpoint import resolve_lightglue_weights
from ..base_model import BaseModel
from .lightglue import LightGlue as LightGlue_


class LightGlue(BaseModel):
    default_conf = {"features": "superpoint", **LightGlue_.default_conf}
    required_data_keys = [
        "view0",
        "keypoints0",
        "descriptors0",
        "view1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        dconf = OmegaConf.to_container(conf)
        dconf["trainable"] = False
        # Resolve the checkpoint to a concrete local path (downloading if needed)
        # so the inner LightGlue loads it via its existing path branch.
        dconf["weights"] = str(resolve_lightglue_weights(dconf.get("weights")))
        self.net = LightGlue_(dconf)
        self.net.eval()
        self.set_initialized()

    def _forward(self, data):
        # required_keys = ["keypoints0", "descriptors0", "keypoints1", "descriptors1"]
        return self.net(data)

    def loss(pred, data):
        raise NotImplementedError

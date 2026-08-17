import numpy as np
import torch

from ...eval.utils import get_matches_scores
from ..base_model import BaseModel
from .nearest_neighbor_matcher import NearestNeighborMatcher


def F_opencv(pts0, pts1, conf):
    import cv2

    try:
        Fm, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            cv2.USAC_MAGSAC,  # decision based on: https://opencv.org/blog/evaluating-opencvs-new-ransacs/
            conf.inlier_threshold,
            conf.success_prob,
            conf.max_iterations,
        )
    except cv2.error:
        Fm, mask = None, None
    return Fm, mask


def F_poselib(pts0, pts1, conf):
    import poselib

    Fm, info = poselib.estimate_fundamental(
        pts0,
        pts1,
        {
            "max_epipolar_error": conf.inlier_threshold,
            "max_iterations": conf.max_iterations,
            "success_prob": conf.success_prob,
        },
    )
    mask = info.pop("inliers") if Fm is not None else None

    return Fm, mask


def robust_F_estimation(data, matches, conf, label=None):
    B, N_kpts, _ = data["keypoints0"].shape

    inliers0_b = torch.zeros((B, N_kpts), dtype=torch.int, device=data["keypoints0"].device)
    outliers0_b = torch.zeros((B, N_kpts), dtype=torch.int, device=data["keypoints0"].device)
    success_b = torch.zeros(B, dtype=torch.bool, device=data["keypoints0"].device)
    F_b = torch.zeros((B, 3, 3), dtype=torch.float, device=data["keypoints0"].device)

    for b in range(B):
        if label is not None and not label[b]:
            m_kpts0, m_kpts1, _ = get_matches_scores(
                data["keypoints0"][b],
                data["keypoints1"][b],
                matches["matches0"][b],
                matches["matching_scores0"][b],
            )

            # skip negative samples
            inliers = torch.ones(m_kpts0.shape[0], dtype=torch.bool, device=m_kpts0.device)
            success = True
        else:
            m_kpts0, m_kpts1, _ = get_matches_scores(
                data["keypoints0"][b],
                data["keypoints1"][b],
                matches["matches0"][b],
                matches["matching_scores0"][b],
            )

            pts0 = m_kpts0.cpu().numpy()
            pts1 = m_kpts1.cpu().numpy()

            if len(pts0) < 16:  # fundamental matrix requires at least 8 points
                Fm, mask = None, None
            else:
                if conf.toolkit == "opencv":
                    Fm, mask = F_opencv(pts0, pts1, conf)
                elif conf.toolkit == "poselib":
                    Fm, mask = F_poselib(pts0, pts1, conf)
                else:
                    raise ValueError(f"Unknown toolkit for robust F estimation: {conf.toolkit}")

            success = Fm is not None and mask is not None
            if success:
                inliers = torch.tensor(mask, dtype=torch.bool, device=m_kpts0.device).squeeze()
                outliers = ~inliers
            else:
                inliers = torch.zeros(m_kpts0.shape[0], dtype=torch.bool, device=m_kpts0.device)
                outliers = ~inliers
                Fm = np.eye(3, dtype=np.float64)
        # like:
        # i, j = 0, 0
        # for i in range(N_kpts):
        #     if matches["matches0"][b, i] > -1:
        #         if inliers[j]:
        #             inliers0_test[i] = matches["matches0"][b, i].item()
        #         j += 1

        matches_b = matches["matches0"][b]  # shape: (N_kpts,)
        valid_mask = matches_b > -1  # valid matches
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)  # positions of valid matches

        # Select only valid matches
        matched_ids = matches_b[valid_mask]  # shape: (num_valid,)

        inliers0 = -1 * torch.ones(N_kpts, dtype=torch.int64, device=m_kpts0.device)
        outliers0 = -1 * torch.ones(N_kpts, dtype=torch.int64, device=m_kpts0.device)

        # Apply inlier mask
        if inliers.numel() > 0:
            valid_inliers = inliers.to(matched_ids.device)
            inlier_indices = valid_indices[valid_inliers]
            inlier_matches = matched_ids[valid_inliers]
            inliers0[inlier_indices] = inlier_matches

        if outliers.numel() > 0:
            valid_outliers = outliers.to(matched_ids.device)
            outlier_indices = valid_indices[valid_outliers]
            outlier_matches = matched_ids[valid_outliers]
            outliers0[outlier_indices] = outlier_matches

        inliers0_b[b] = (
            inliers0  # indices of matched keypoints in image 1 THAT SURVIVED ROBUST ESTIMATION, -1 for non-matches and outliers
        )
        outliers0_b[b] = (
            outliers0  # indices of matched keypoints in image 1 THAT WERE REJECTED BY ROBUST ESTIMATION, -1 for non-matches and inliers
        )
        success_b[b] = success
        F_b[b] = torch.from_numpy(Fm).to(m_kpts0.device)

    return inliers0_b, outliers0_b, success_b, F_b


class RobustNearestNeighborMatcher(BaseModel):
    default_conf = {
        "conf_nn_matcher": {
            "ratio_thresh": None,
            "distance_thresh": None,
            "mutual_check": True,
            "loss": None,
        },
        "toolkit": "opencv",
        "inlier_threshold": 0.5,
        "max_iterations": 10000,
        "success_prob": 0.99,
    }
    required_data_keys = ["descriptors0", "descriptors1", "keypoints0", "keypoints1"]

    def _init(self, conf):
        self.nn_matcher = NearestNeighborMatcher(conf["conf_nn_matcher"])

    def _forward(self, data):
        matches = self.nn_matcher(data)

        inliers0_b, outliers0_b, success_b = robust_F_estimation(data, matches, self.conf)

        matches["inliers0"] = inliers0_b
        matches["outliers0"] = outliers0_b
        matches["success"] = success_b

        return matches

    def loss(self, pred, data):
        raise NotImplementedError

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.epipolar import sampson_epipolar_distance
from omegaconf import OmegaConf

from ... import logger
from ...visualization.visualize_batch import plot_match_figures
from ..matchers.robust_nearest_neighbor_matcher import robust_F_estimation


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights  # element-wise weighting of log-probabilities

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)  # count of non-matchable (dustbin)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)  # count of non-matchable (dustbin)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)  # count of positive matches (inside the m x n block)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)  # summed negative log-likelihood of positive matches, normalized by count

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)  # summed negative log-likelihood of non-matchable in set 0
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)  # summed negative log-likelihood of non-matchable in set 1

    nll_neg = (nll_neg0 + nll_neg1) / (
        num_neg0 + num_neg1
    )  # averaged negative log-likelihood of non-matchable, normalized by count

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(log_assignment, weights, gamma=self.conf.gamma_f)
        nll = (
            self.conf.nll_balancing * nll_pos
            + (1 - self.conf.nll_balancing) * nll_neg  # this is equation 11 (somewhat)
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        """Despite its name, this function actually only creates weights for NLL loss.
        It has ones where there are ground truth matches, or a keypoint is marked unmatchable (dustbin).
        It has zeros elsewhere."""

        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return weights


class HuberKernel:
    def __init__(self, delta=1.0, threshold=1.0, outlier_penalty=-0.1):
        self.delta = delta
        self.threshold = threshold
        self.outlier_penalty = outlier_penalty

        # do Huber sanity check
        logger.info(f"Huber kernel initialized with delta={self.delta}, threshold={self.threshold}")
        point_of_intersection = (2 * self.delta**2) ** 0.5
        logger.info(f"Intersection with x-axis at {point_of_intersection}")
        assert point_of_intersection <= self.threshold, (
            "Huber kernel threshold too small! This would lead to positive rewards for outliers."
        )

    def __call__(self, d):
        val = 1 - 0.5 * (d / (self.delta + 1e-8)) ** 2
        val[d > self.threshold] = self.outlier_penalty
        return val


class CauchyKernel:
    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, d):
        val = 1 / (1 + (d / (self.delta + 1e-8)) ** 2)
        return val


def get_kernel_fn(kernel_conf):
    if kernel_conf["type"] == "huber":
        return HuberKernel(
            delta=kernel_conf["delta"],
            threshold=kernel_conf["threshold"],
            outlier_penalty=kernel_conf["outlier_penalty"],
        )
    elif kernel_conf["type"] == "cauchy":
        return CauchyKernel(delta=kernel_conf["delta"])
    else:
        raise ValueError(f"Unknown kernel type: {kernel_conf['type']}")


class RLLoss(nn.Module):
    default_conf = {
        "use_baseline": False,
        "fp_penalty": None,  # penalty for false positives (not finding a match in positive case, or finding a match in negative case)
        "kp_penalty": None,  # penalty for labeling a keypoint as non-matchable
        "estimator": {
            "toolkit": "opencv",
            "inlier_threshold": 0.5,
            "max_iterations": 10000,
            "success_prob": 0.99,
        },
    }
    alpha = None

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        # self.loss_fn = self.rl_loss
        self.kernel_fn = get_kernel_fn(self.conf.kernel)

    def update_alpha(self, alpha):
        self.alpha = alpha

    def plausible_matches(self, data):
        """Computes plausible matches by filtering the predicted matches based on a estimated fundamental matrix."""

        matches = {
            "matches0": data["matches0"],
            "matching_scores0": data["matching_scores0"],
            "matches1": data["matches1"],
            "matching_scores1": data["matching_scores1"],
        }

        inliers0, outliers0, success, Fm = robust_F_estimation(data, matches, self.conf.estimator, data["label"])

        return inliers0, outliers0, success, Fm

    def pairwise_sampson_distance(self, A_kpts, B_kpts, Fm):
        """
        Args:
            A_kpts: (B, N, 2)
            B_kpts: (B, M, 2)
            Fm: (B, 3, 3) fundamental matrices
        Returns:
            (B, N, M) Sampson distances
        """
        B, N, _ = A_kpts.shape
        M = B_kpts.shape[1]

        # Broadcast to all pairs
        A_exp = A_kpts[:, :, None, :].expand(B, N, M, 2)  # (B, N, M, 2)
        B_exp = B_kpts[:, None, :, :].expand(B, N, M, 2)  # (B, N, M, 2)

        # Flatten for kornia (expects (B, N, 2))
        A_flat = A_exp.reshape(B, N * M, 2)
        B_flat = B_exp.reshape(B, N * M, 2)

        # Sampson distance
        dist_flat = sampson_epipolar_distance(A_flat, B_flat, Fm)  # (B, N*M)

        return dist_flat.reshape(B, N, M)

    def forward(self, pred, data, rewards=None):
        log_assignment = pred["log_assignment"]
        log_assignment_match = log_assignment[:, :-1, :-1]
        log_non_matchable_A = log_assignment[:, :-1, -1]
        log_non_matchable_B = log_assignment[:, -1, :-1]
        similarity = pred["similarity"]

        if rewards is None:
            # filter matches based on geometric plausibility
            # reward only matches that are geometrically plausible
            inliers0, outliers0, success, Fm = self.plausible_matches(data)
            data["F_inliers0"] = inliers0
            data["F_outliers0"] = outliers0
            data["F_success"] = success
            data["Fm_estimated"] = Fm

            rewards = self.compute_dense_reward(data)  # [B, M+1, N+1]

        if False:
            pred = data
            plot_match_figures(pred, data, 3, do_plot_keypoints=True, do_plot_matches=True)

        with torch.no_grad():
            sample_p_A = F.softmax(similarity, 2)
            sample_p_B = F.softmax(similarity.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
            sample_p = sample_p_A * sample_p_B  # [B, M, N]

            # Baseline variance reduction
            if self.conf.use_baseline:
                baseline = (sample_p * rewards).sum(dim=(1, 2), keepdim=True) / sample_p.sum(
                    dim=(1, 2), keepdim=True
                ).clamp(min=1e-8)
                advantages = rewards - baseline
            else:
                advantages = rewards

        reinforce = (sample_p * advantages * log_assignment_match).sum(dim=(1, 2))

        loss_stats = {}

        if self.conf.kp_penalty is not None:
            # soft-count of keypoints assigned to dustbin
            dustbin_prob_sum_A = log_non_matchable_A.exp().sum(dim=1)
            dustbin_prob_sum_B = log_non_matchable_B.exp().sum(dim=1)
            dustbin_prob_sum = dustbin_prob_sum_A + dustbin_prob_sum_B
            kp_penalty = self.conf.kp_penalty * dustbin_prob_sum

            loss = -reinforce + kp_penalty

            loss_stats["kp_penalty"] = kp_penalty
        else:
            loss = -reinforce

        # statistics
        sum_rewards = rewards.sum(dim=(1, 2))
        sum_rewards_pos = sum_rewards[data["label"]]

        # loss = -(log_assignment * rewards).mean(dim=(1, 2))

        return (
            loss,
            rewards,
            {
                "rl_loss": loss,
                "reinforce": -reinforce,
                "sum_rewards": sum_rewards,
                "sum_rewards_pos": sum_rewards_pos,
                **({"baseline": baseline.squeeze()} if self.conf.use_baseline else {}),
                **loss_stats,
            },
        )

    def compute_dense_reward(self, data):
        """
        Compute dense reward matrix based on ground truth matches.
        data: dictionary containing ground truth matches and other relevant information
        Returns a reward matrix of the same shape as log_assignment.
        """

        m, n = data["matches0"].size(-1), data["matches1"].size(-1)
        B = data["matches0"].size(0)

        default_reward = torch.ones(data["label"].shape, device=data["label"].device) * self.conf.inlier_reward
        default_penalty = torch.ones(data["label"].shape, device=data["label"].device) * self.conf.outlier_penalty

        dense_reward = torch.zeros((B, m, n), device=data["matches0"].device)  # B, M, N

        if self.conf.fp_penalty is not None:
            # set penalty for not finding a match (positive case) or finding a match (negative case)
            penalty = self.conf.fp_penalty * self.alpha * default_reward  # small penalty scaled by alpha

            dense_reward[:, :, :] = penalty.view(B, 1, 1)  # for all possible matches

        if self.conf["reward_based_on_sampson_distance"]:
            # compute Sampson distances for inlier matches
            Fm = data["Fm_estimated"]

            all_matches0 = data["matches0"]  # B, M

            m_kpts0_batched = torch.stack([data["keypoints0"][b, :] for b in range(B)], dim=0)  # B, num_inliers, 2
            m_kpts1_batched = torch.stack(
                [data["keypoints1"][b, all_matches0[b]] for b in range(B)], dim=0
            )  # B, num_inliers, 2

            # attention: also calculates distances for pairs that are not actually matched, but these will be masked out later anyway, so this is not a problem

            rewards_based_on_distance = self.kernel_fn(
                sampson_epipolar_distance(m_kpts0_batched, m_kpts1_batched, Fm, squared=True)
            )  # B, num_inliers

            # Batched
            # Create mask for valid matches
            valid_mask = all_matches0 != -1  # (B, m)

            # Get batch and row indices
            batch_idx = torch.arange(B, device=all_matches0.device)[:, None].expand(B, m)
            row_idx = torch.arange(m, device=all_matches0.device)[None, :].expand(B, m)

            # Clamp column indices to valid range (invalid matches will be masked anyway)
            col_idx = all_matches0.clamp(min=0)

            # Batch assignment using advanced indexing
            dense_reward[batch_idx[valid_mask], row_idx[valid_mask], col_idx[valid_mask]] = rewards_based_on_distance[
                valid_mask
            ]

        else:  # discrete rewards
            # matches
            matches0_inlier = data[
                "F_inliers0"
            ]  # B, M --- indices of matched keypoints in image 1 THAT SURVIVED ROBUST ESTIMATION, -1 for non-matches and outliers
            matches0_outlier = data[
                "F_outliers0"
            ]  # B, M --- indices of matched keypoints in image 1 THAT WERE REJECTED BY ROBUST ESTIMATION, -1 for non-matches and inliers

            # print((matches0_inlier!=-1).sum(dim=(1)))
            # print((matches0_outlier!=-1).sum(dim=(1)))

            # LIKE:
            # for b in range(B):
            #     print(f"Batch {b}")
            #     for i in range(m):
            #         if matches0[b, i] != -1:
            #             dense_reward[b, i, matches0[b, i]] = reward[b]
            # INFO: stepping through matches1 the same way is not required, as it would just address the same entries again
            # as the matching information is symmetric / redundant

            # batch indices expanded
            batch_idx = torch.arange(B, device=matches0_inlier.device)[:, None].expand(B, m)

            # if self.conf.use_baseline:
            #     baseline_inlier = (data["gt_inliers1"] > -1).sum(dim=1)
            #     lightglue_inlier = (matches0 != -1).sum(dim=1)

            #     reward = (lightglue_inlier - baseline_inlier) / (lightglue_inlier + 1e-6)

            # --- first loop equivalent ---
            # reward for inlier matches
            mask0 = matches0_inlier != -1
            dense_reward[
                batch_idx[mask0],
                torch.arange(m, device=matches0_inlier.device).expand(B, m)[mask0],
                matches0_inlier[mask0],
            ] = default_reward[batch_idx[mask0]]

            # penalty for outlier matches
            mask0 = matches0_outlier != -1
            dense_reward[
                batch_idx[mask0],
                torch.arange(m, device=matches0_outlier.device).expand(B, m)[mask0],
                matches0_outlier[mask0],
            ] = default_penalty[batch_idx[mask0]]

            # if self.conf.use_dustbin_rewards:
            #     # # Dustbin rewards
            #     unmatched0 = data["unmatched0"]
            #     unmatched1 = data["unmatched1"]

            #     # LIKE:
            #     # for b in range(B):
            #     #     for i in range(m):
            #     #         if unmatched0[b, i]:
            #     #             dense_reward[b, i, -1] = -reward[b]
            #     #     for j in range(n):
            #     #         if unmatched1[b, j]:
            #     #             dense_reward[b, -1, j] = -reward[b]

            #     # Expand reward for broadcasting
            #     reward = -0.5 * reward[:, None]  # shape (B, 1)

            #     # Set dense_reward[b, i, -1] = -reward[b] where unmatched0[b, i] is True
            #     dense_reward[:, :-1, -1] = torch.where(unmatched0, reward, dense_reward[:, :-1, -1])

            #     # Set dense_reward[b, -1, j] = -reward[b] where unmatched1[b, j] is True
            #     dense_reward[:, -1, :-1] = torch.where(unmatched1, reward, dense_reward[:, -1, :-1])

        return dense_reward

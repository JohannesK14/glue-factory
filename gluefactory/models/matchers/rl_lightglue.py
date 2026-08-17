import torch
from omegaconf import OmegaConf
from torch import nn

from ..utils.losses import RLLoss
from .lightglue import LightGlue, filter_matches, matcher_metrics, normalize_keypoints, sigmoid_log_double_softmax


def filter_matches_with_dustbin(scores: torch.Tensor, th: float):
    """Extended match filtering with dustbin-aware argmax.

    Procedure (option 3):
      1. For each keypoint, take argmax over ALL targets including dustbin.
      2. If dustbin wins, mark unmatched immediately (cannot become a match).
      3. Among remaining candidates, enforce mutual consistency (reciprocal argmax) still ignoring dustbin.
      4. Apply threshold on the (exp) score, then finalize indices.

    Args:
        scores: Log assignment matrix [B, M+1, N+1] (last row/col are dustbins).
        th: Threshold on mutual match probability (after exp of log-score).

    Returns:
        m0, m1: (B,M)/(B,N) mutual match indices or -1.
        mscores0, mscores1: (B,M)/(B,N) probabilities (0 if invalid / below th).
        unmatched0, unmatched1: (B,M)/(B,N) bool masks where dustbin had global argmax.
        dustbin_scores0, dustbin_scores1: (B,M)/(B,N) probabilities of assigning each keypoint to dustbin.
    """
    B, Mp1, Np1 = scores.shape
    M, N = Mp1 - 1, Np1 - 1

    # Global argmax including dustbin for each keypoint.
    full_max0 = scores[:, :M, :].max(2)  # over N + 1 (includes dustbin col at index N)
    full_max1 = scores[:, :, :N].max(1)  # over M + 1 (includes dustbin row at index M)

    # Dustbin selection masks.
    unmatched0 = full_max0.indices == N  # dustbin column index
    unmatched1 = full_max1.indices == M  # dustbin row index

    # Candidate partner indices (only valid if not unmatched).
    partner0 = full_max0.indices.clamp(max=N - 1)
    partner1 = full_max1.indices.clamp(max=M - 1)
    cand0 = ~unmatched0
    cand1 = ~unmatched1

    # Mutual consistency among candidates.
    # For keypoint i in set0 with partner j, require partner j is candidate and selects i back.
    indices0 = torch.arange(M, device=scores.device)[None, :]
    indices1 = torch.arange(N, device=scores.device)[None, :]

    # partner1.gather(1, partner0) gives partner1[j] for each selected j.
    mutual0 = cand0 & cand1.gather(1, partner0) & (indices0 == partner1.gather(1, partner0))
    mutual1 = cand1 & cand0.gather(1, partner1) & (indices1 == partner0.gather(1, partner1))

    # Scores (probabilities) for candidate matches (exp of log-score at the chosen pair).
    pair_scores0 = full_max0.values.exp()  # includes dustbin exp for unmatched; will be zeroed later
    pair_scores1 = full_max1.values.exp()

    # Construct mscores with mutual constraint.
    mscores0 = torch.where(mutual0, pair_scores0, pair_scores0.new_zeros(()).expand_as(pair_scores0))
    mscores1 = torch.where(mutual1, pair_scores1, pair_scores1.new_zeros(()).expand_as(pair_scores1))

    # Threshold filtering.
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & (mscores1 > th) & valid0.gather(1, partner1)

    # Final indices (-1 where not valid or unmatched).
    m0 = torch.where(valid0, partner0, partner0.new_full(partner0.shape, -1))
    m1 = torch.where(valid1, partner1, partner1.new_full(partner1.shape, -1))

    # Zero out scores where invalid after threshold.
    mscores0 = torch.where(valid0, mscores0, mscores0.new_zeros(()).expand_as(mscores0))
    mscores1 = torch.where(valid1, mscores1, mscores1.new_zeros(()).expand_as(mscores1))

    # Dustbin probabilities (exp of log-prob) for analysis.
    dustbin_scores0 = scores[:, :M, -1].exp()
    dustbin_scores1 = scores[:, -1, :N].exp()

    return (
        m0,
        m1,
        mscores0,
        mscores1,
        unmatched0,
        unmatched1,
        dustbin_scores0,
        dustbin_scores1,
    )


def augment_matches_with_nn(
    scores: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
    mscores0: torch.Tensor,
    mscores1: torch.Tensor,
    min_matches: int,
    threshold: float = 0.0,
):
    """
    Augment existing mutual matches with additional NN matches if needed.

    This function implements a hybrid matching strategy:
    1. Keeps all existing mutual matches (from filter_matches)
    2. If mutual matches < min_matches, fills up with non-mutual NN matches
    3. Additional NN matches only update m0 (A→B direction), not m1

    Args:
        scores: Log assignment matrix [B, M+1, N+1] (last row/col are dustbins)
        m0, m1: Existing mutual matches from filter_matches [B, M] and [B, N]
        mscores0, mscores1: Existing match scores [B, M] and [B, N]
        min_matches: Target minimum number of matches per image pair
        threshold: Minimum score (probability) for additional NN matches

    Returns:
        m0, m1, mscores0, mscores1: Augmented versions (only m0 and mscores0 modified)
    """
    B, Mp1, Np1 = scores.shape
    M, N = Mp1 - 1, Np1 - 1

    # Clone to avoid modifying originals
    m0 = m0.clone()
    mscores0 = mscores0.clone()

    # Count existing mutual matches per batch
    num_mutual0 = (m0 >= 0).sum(dim=1)  # [B]

    # Find batches that need augmentation
    needs_augment = num_mutual0 < min_matches  # [B] bool mask

    if not needs_augment.any():
        return m0, m1, mscores0, mscores1

    # Process each batch that needs augmentation
    for b in range(B):
        if not needs_augment[b]:
            continue

        # How many additional matches needed?
        needed = min_matches - num_mutual0[b].item()

        # Get all NN matches from A→B (excluding dustbin)
        nn_scores, nn_indices = scores[b, :M, :N].max(dim=1)  # [M]
        nn_probs = nn_scores.exp()

        # Mask out already matched keypoints
        already_matched = m0[b] >= 0
        nn_probs = nn_probs.clone()
        nn_probs[already_matched] = -float("inf")

        # Apply threshold
        valid_nn = nn_probs > threshold
        nn_probs[~valid_nn] = -float("inf")

        # Select top-k additional matches
        k = min(needed, M - already_matched.sum().item())
        if k <= 0:
            continue

        topk_probs, topk_kp_indices = nn_probs.topk(k, sorted=False)

        # Filter out invalid (-inf) entries
        valid_topk = topk_probs > -float("inf")
        topk_kp_indices = topk_kp_indices[valid_topk]
        topk_probs = topk_probs[valid_topk]

        # Add these to m0 and mscores0 (Option A: only A→B direction)
        m0[b, topk_kp_indices] = nn_indices[topk_kp_indices]
        mscores0[b, topk_kp_indices] = topk_probs

    return m0, m1, mscores0, mscores1


class MatchAssignment(nn.Module):
    def __init__(self, dim: int, softmax_temperature: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.softmax_temperature = softmax_temperature

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        sim = sim * self.softmax_temperature  # temperature scaling
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)

    def set_softmax_temperature(self, temperature: float):
        self.softmax_temperature = temperature

    def get_softmax_temperature(self):
        return self.softmax_temperature


class RL_LightGlue(LightGlue):
    # Start from LightGlue defaults, only override what changes.
    default_conf = {
        **LightGlue.default_conf,
        "name": "rl_lightglue",
        "loss": {
            **LightGlue.default_conf["loss"],
            "fn": "rl",  # switch loss function
            "use_baseline": False,
            "fp_penalty": None,
            "kp_penalty": None,
        },
        "min_matches": 0,  # If >0, augment mutual matches with NN to reach this target
        "nn_threshold": 0.0,  # Minimum score (probability) for additional NN matches
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    # url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        n, d = conf.n_layers, conf.descriptor_dim

        # overwrite assignment modules
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.loss_fn = RLLoss(conf.loss)

    def set_softmax_temperature_matcher(self, temperature: float):
        for la in self.log_assignment:
            la.set_softmax_temperature(temperature)

    def get_softmax_temperature_matcher(self):
        return self.log_assignment[-1].get_softmax_temperature()

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.transformers[i],
                    desc0,
                    desc1,
                    encoding0,
                    encoding1,
                    use_reentrant=False,  # Recommended by torch, default was True
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

            # only for eval
            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        # m0, m1, mscores0, mscores1, unmatched0, unmatched1, unscore0, unscore1 = filter_matches_with_dustbin(
        #     scores, self.conf.filter_threshold
        # )
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        # Optionally augment with NN matches to reach minimum target
        if self.conf.min_matches > 0:
            m0, m1, mscores0, mscores1 = augment_matches_with_nn(
                scores, m0, m1, mscores0, mscores1, min_matches=self.conf.min_matches, threshold=self.conf.nn_threshold
            )

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
            # "unmatched0": unmatched0,
            # "unmatched1": unmatched1,
            # "unmatched_scores0": unscore0,
            # "unmatched_scores1": unscore1,
        }

        return pred

    # override loss function from original LightGlue
    def loss(self, pred, data):
        def loss_params(pred, i):
            # berechnet log assignment für den entsprechenden Layer i
            # mit MatchAssignment.forward() basierend auf den beiden Descriptoren
            la, sim = self.log_assignment[i](pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i])
            return {
                "log_assignment": la,
                "similarity": sim,
            }

        sum_weights = 1.0
        rll, rewards_last_layer, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": rll, "last": rll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].clone().detach().exp()[:, :-1].sum(2).mean(1)
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            rll, _, _ = self.loss_fn(params_i, data, rewards=rewards_last_layer)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + rll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights

        # confidences
        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = RL_LightGlue

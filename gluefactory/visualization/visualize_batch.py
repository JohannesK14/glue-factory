import matplotlib.pyplot as plt
import numpy as np
import torch

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches


def make_match_figures(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append([view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)])
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0) for i in range(n_pairs)]

    return {"matching": fig}


def plot_match_figures(pred_, data_, n_pairs=2, do_plot_keypoints=False, do_plot_matches=False):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    F_inliers0 = pred["F_inliers0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (F_inliers0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append([view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)])
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = F_inliers0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True, dpi=300)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    if do_plot_keypoints:
        [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    if do_plot_matches:
        [
            plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0, subsample=10)
            for i in range(n_pairs)
        ]

    plt.savefig("match_visualization.png")


def plot_matches_based_on_sampson_distance(data, batch_idx=0, T=0.001):
    sd = data["sampson_distance"][batch_idx, :, :]

    indices = torch.where(sd < T)
    values = sd[indices]

    val_idx_tuples = list(
        zip(values.tolist(), list(zip(indices[0].tolist(), indices[1].tolist(), strict=False)), strict=False)
    )
    val_idx_tuples.sort(key=lambda x: x[0])

    # get all keypoint locations below a certain sampson distance threshold
    kpts_sampson_0 = []
    kpts_sampson_1 = []
    for v, (i, j) in val_idx_tuples:
        if v < T:
            kpts_sampson_0.append(data["keypoints0"][batch_idx, i].cpu().numpy())
            kpts_sampson_1.append(data["keypoints1"][batch_idx, j].cpu().numpy())

    print(f"Number of keypoint pairs with sampson distance < {T}: {len(kpts_sampson_0)}")

    kpts_sampson_0 = np.array(kpts_sampson_0)
    kpts_sampson_1 = np.array(kpts_sampson_1)

    img0 = data["view0"]["image"][batch_idx].permute(1, 2, 0).cpu().numpy()
    img1 = data["view1"]["image"][batch_idx].permute(1, 2, 0).cpu().numpy()

    images = [[img0, img1]]
    kpts = [[kpts_sampson_0, kpts_sampson_1]]
    matches = [(kpts_sampson_0, kpts_sampson_1)]

    _fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    plot_keypoints(kpts[0], axes=axes[0], colors="royalblue")
    plot_matches(*matches[0], color=None, axes=axes[0], a=0.5, lw=1.0, ps=0.0)
    plt.savefig("sampson_distance_viz.png")

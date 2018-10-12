import os
import cv2
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from functools import partial

from helpers.utils import ensure_dir
from helpers.multi_process import map_in_pool


def vis_2d_heatmap(heatmap, figsize=None, gt_point_list=None, pred_point_list=None, filename=None, interp=None, vmax=None, vmin=None):
    if figsize is None:
        figsize = (10, 10)
    plt.figure(figsize=figsize, dpi=100)
    plt.set_cmap('jet')
    if vmax is None and vmin is None:
        plt.imshow(heatmap, interpolation='nearest' if interp is None else interp)
    else:
        plt.imshow(heatmap, interpolation='nearest' if interp is None else interp, vmax=vmax, vmin=vmin)
    if gt_point_list is not None:
        xs = [p[0] for p in gt_point_list if np.sum(p) > 0]
        rs = [p[1] for p in gt_point_list if np.sum(p) > 0]
        plt.scatter(x=rs, y=xs, s=15, c='white', marker='x')  # white point gt
    if pred_point_list is not None:
        xs = [p[0] for p in pred_point_list if np.sum(p) > 0]
        rs = [p[1] for p in pred_point_list if np.sum(p) > 0]
        plt.scatter(x=rs, y=xs, s=15, c='grey', marker='x')  # gery point pred
    plt.savefig("image.png" if filename is None else filename)
    plt.clf()
    plt.close()


def vis_3d_heatmap_distributor_y(heatmap, Y):
    for y in range(Y):
        yield heatmap[:, y, :], y


def vis_3d_heatmap_worker_y(record, verbose, file_path):
    heatmap, i = record
    if verbose:
        print("y: {}".format(i))
    vis_2d_heatmap(heatmap=heatmap, filename=os.path.join(file_path, "y_{:03d}".format(i)))


def vis_3d_heatmap_distributor_x(heatmap, X):
    for x in range(X):
        yield heatmap[x, :, :], x


def vis_3d_heatmap_worker_x(record, verbose, file_path):
    heatmap, i = record
    if verbose:
        print("x: {}".format(i))
    vis_2d_heatmap(heatmap=heatmap, filename=os.path.join(file_path, "x_{:03d}".format(i)))


def vis_3d_heatmap(heatmap, file_path):
    assert len(heatmap.shape) == 3
    X, Y, Z = heatmap.shape

    ensure_dir(file_path)

    # use multi process when heat map is large.
    if max(Y, X) > 16:

        _ = map_in_pool(
            fn=partial(vis_3d_heatmap_worker_y, verbose=True, file_path=file_path),
            data=vis_3d_heatmap_distributor_y(heatmap, Y),
            single_process=False,
            verbose=True,
            nr_pro=8
        )

        _ = map_in_pool(
            fn=partial(vis_3d_heatmap_worker_x, verbose=True, file_path=file_path),
            data=vis_3d_heatmap_distributor_x(heatmap, X),
            single_process=False,
            verbose=True,
            nr_pro=8
        )

    else:

        # visualize horizontal plane
        for y in range(Y):
            vis_2d_heatmap(heatmap=heatmap[:, y, :], filename=os.path.join(file_path, "y_{}".format(y)))

        # visualize vertical plane
        for x in range(X):
            vis_2d_heatmap(heatmap=heatmap[x, :, :], filename=os.path.join(file_path, "x_{}".format(x)))

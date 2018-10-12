import os
import cv2
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from functools import partial
import gc

import models.RPN.config as rpn_config
import config
from helpers.tqdm import tqdm
from helpers.utils import ensure_dir
from helpers.multi_process import map_in_pool
from helpers.visualization import vis_2d_heatmap
from helpers.online_matric import pose_dist
from models.RCNN import config as pose_config

from IPython import embed


def concat_image(read_paths, target_path):
    images = [cv2.imread(p) for p in read_paths]
    image_concat = np.concatenate(images, axis=1)
    cv2.imwrite(target_path, image_concat)


def draw_poses_with_gt(pose_list, gt_pose_list, save_file_path):

    # check if image exists
    if os.path.exists(save_file_path + ".png"):
        return

    fig = plt.figure(figsize=(8.8, 6.6), dpi=100)
    ax = Axes3D(fig)

    ax.text2D(0.05, 0.95, os.path.basename(save_file_path), transform=ax.transAxes)

    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.set_zlim([0, 60])

    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('R', fontsize=20)
    ax.set_zlabel('Y', fontsize=20)

    # prediction
    for idx_pose, pose in enumerate(pose_list):

        xs = [pose[i][0] for i in range(14)]
        # TODO: why upside down?
        ys = [60 - pose[i][1] for i in range(14)]
        zs = [pose[i][2] for i in range(14)]

        # put vertical axis y to third axis of matplotlib
        ax.scatter(xs, zs, ys, s=10, c='red', marker='o')

        # draw skeleton lines
        for (i, j) in pose_config.KPT_LINE_IDX:
            ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]], color='red', linestyle='-', linewidth=1)

        # add proposals index
        ax.text(x=xs[-1], y=zs[-1], z=ys[-1], s="{}".format(idx_pose))

    # ground truth
    for idx_pose, pose in enumerate(gt_pose_list):
        # skip gt poses
        if np.sum(pose) == 0:
            continue

        # add offset to x axis to seperate gt from pred.
        xs = [pose[i][0] - 10 for i in range(14) if pose[i][2] > 0]
        # TODO: why upside down? It is designed upsided-down.
        ys = [60 - pose[i][1] for i in range(14) if pose[i][2] > 0]
        zs = [pose[i][2] for i in range(14) if pose[i][2] > 0]

        # print("xs: {}".format(xs))
        # print("ys: {}".format(ys))
        # print("zs: {}".format(zs))

        # put vertical axis y to third axis of matplotlib
        ax.scatter(xs, zs, ys, s=10, c='blue', marker='o')

        # draw skeleton lines
        for (i, j) in pose_config.KPT_LINE_IDX:
            if pose[i][2] < 0 or pose[j][2] < 0:
                continue
            ax.plot([pose[i][0] - 10, pose[j][0] - 10], [pose[i][2], pose[j][2]], [60 - pose[i][1], 60 - pose[j][1]],
                    color='blue', linestyle='-', linewidth=1)

    # # put device on (100, -20, 0)
    # ax.scatter(xs=[100], ys=[0], zs=[20], s=20, c='green', marker='x')
    # ax.plot(xs=[100, 100], ys=[0, 0], zs=[20, 0], color='green', linestyle="--", linewidth=2)

    # save image from 2 angles
    ax.view_init(elev=30)
    plt.savefig(save_file_path + "_l.png")
    ax.view_init(azim=-150, elev=30)
    plt.savefig(save_file_path + "_r.png")

    # clear image.
    plt.cla()

    # concate left right image.
    concat_image([save_file_path + "_l.png", save_file_path + "_r.png"], save_file_path + ".png")
    os.remove(save_file_path + "_l.png")
    os.remove(save_file_path + "_r.png")


# ['exp0606-2-001/006', 'exp0607-2-001/006', 'exp0607-2-002/006', 'exp0613-2-001/006']

def _worker_draw_poses_with_gt(record, verbose=False):
    if verbose and record[3] % 10 == 0:
        print("idx: {}".format(record[3]))
    draw_poses_with_gt(record[0], record[1], record[2])


def gen_3d_pose_vis_image(pred_pose, gt_pose, file_dir, base_time, verbose=True):
    """
    Generate 3d pose visualization images for a time window.
    :param pred_pose:
    :param gt_pose:
    :param heatmap:
    :param proposal:
    :param filename:
    :return:
    """
    ensure_dir(file_dir)
    time_range = len(pred_pose)

    def yield_record():
        for t in tqdm(range(time_range)):
            yield pred_pose[t], gt_pose[t], os.path.join(file_dir, "gt_pred_{:05d}".format(base_time + t)), t

    map_in_pool(
        fn=partial(_worker_draw_poses_with_gt, verbose=verbose),
        data=yield_record(),
        single_process=False,
        nr_pro=12,
        verbose=verbose
    )

    gc.collect()


def vis_2d_heatmap_ver(heatmap_list, filename, gt_point_list, pred_point_list, n_max_prop):
    fig, axeslist = plt.subplots(ncols=2, nrows=int(n_max_prop / 2), figsize=(16, 8), dpi=100)
    for idx, hmap in enumerate(heatmap_list):
        axeslist.ravel()[idx].imshow(hmap)
        if pred_point_list[idx][0] > 0 and pred_point_list[idx][1] > 0:
            axeslist.ravel()[idx].scatter(
                y=[pred_point_list[idx][0]], x=[pred_point_list[idx][1]], c='grey', marker='x', s=25)
        if gt_point_list[idx][0] > 0 and gt_point_list[idx][1] > 0:
            axeslist.ravel()[idx].scatter(
                y=[gt_point_list[idx][0]], x=[gt_point_list[idx][1]], c='white', marker='x', s=25)
        # axeslist.ravel()[idx].set_title("prop_{}".format(idx))
        # axeslist.ravel()[idx].set_axis_off()
    plt.tight_layout()  # optional
    plt.set_cmap('jet')
    plt.savefig("image.png" if filename is None else filename)
    plt.clf()
    plt.close()


def _gen_pose_heatmap_per_t_kpt(record, file_dir, base_time, image_tag, verbose=False):
    t, idx_kpt, proposals_t, heatmap_h_t, heatmap_v_t, kpt_pred_t, kpt_gt_t = record

    if verbose and t % 10 == 0:
        print("time: {}, kpt_idx: {}".format(t, idx_kpt))

    min_nr_prop = 6  # minimum nr proposal
    scale = 4  # ratio to zoom in heatmap
    kpt_name = pose_config.KPT_NAMES[idx_kpt]  # keypoint name

    # heatmap space size
    l_x, l_y, l_r = config.HOR_SHAPE[1] * scale, config.VER_SHAPE[1] * scale, config.HOR_SHAPE[2] * scale

    # hor view use only one image. ver image separated.
    image_h = np.zeros((l_x, l_r), dtype=np.uint8)
    image_v_list = []

    # save pred and gt points
    pred_pose_list, gt_pose_list = [], []

    # sort proposals by r_min
    prop_r_mins = np.array([p[1] for p in proposals_t])
    prop_r_mins_idx = np.argsort(prop_r_mins)

    # for each proposals
    for idx_prop in prop_r_mins_idx:
        prop = proposals_t[idx_prop]

        save_image_path = os.path.join(
            file_dir,
            image_tag.format(time=base_time + t, kpt_name=kpt_name.replace(" ", "-").lower(),
                             h_v="v", prop="{}".format(idx_prop))
        )
        if os.path.exists(save_image_path):
            continue

        x_min, r_min, x_max, r_max = prop[0:4] * scale
        x_min, r_min, x_max, r_max = int(x_min), int(r_min), int(x_max), int(r_max)

        # resize
        h_hor = heatmap_h_t[idx_prop][idx_kpt]
        h_ver = heatmap_v_t[idx_prop][idx_kpt]

        # rescaling to 0-255
        h_hor = (h_hor - np.min(h_hor)) / (np.max(h_hor) - np.min(h_hor)) * 255
        h_ver = (h_ver - np.min(h_ver)) / (np.max(h_ver) - np.min(h_ver)) * 255
        h_hor = h_hor.astype(np.uint8)
        h_ver = h_ver.astype(np.uint8)

        # resize heatmap
        h, w = x_max - x_min, r_max - r_min
        # NOTE: cv2 resize: width comes first
        h_hor = cv2.resize(h_hor, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        h_ver = cv2.resize(h_ver, dsize=(w, l_y), interpolation=cv2.INTER_LINEAR)

        # paste the heatmap to image
        image_h[x_min:x_min + h_hor.shape[0], r_min:r_min + h_hor.shape[1]] = h_hor
        this_image_v = np.zeros((l_y, l_r), dtype=np.uint8)
        this_image_v[:, r_min:r_min + h_ver.shape[1]] = h_ver
        image_v_list.append(this_image_v)

        # prepare pose pred and gt for hor image
        pred_pose_list.append(kpt_pred_t[idx_prop])
        gt_pose_list.append(kpt_gt_t[idx_prop])

    # save hor
    save_image_path = os.path.join(file_dir, image_tag.format(
        time=base_time + t, kpt_name=kpt_name.replace(" ", "-").lower(), h_v="h"))
    if not os.path.exists(save_image_path):
        # prepare hor kpt points
        pred_kpts = [(p[0] * scale, p[2] * scale) for p in pred_pose_list]
        gt_kpts = [(p[0] * scale, p[2] * scale) for p in gt_pose_list]
        vis_2d_heatmap(heatmap=image_h, gt_point_list=gt_kpts, pred_point_list=pred_kpts, filename=save_image_path,
                       figsize=(8, 8))
    del image_h

    save_image_path = os.path.join(file_dir, image_tag.format(
        time=base_time + t, kpt_name=kpt_name.replace(" ", "-").lower(), h_v="v"))
    if not os.path.exists(save_image_path):
        # prepare ver kpt points
        pred_kpts = [(p[1] * scale, p[2] * scale) for p in pred_pose_list]
        gt_kpts = [(p[1] * scale, p[2] * scale) for p in gt_pose_list]
        # pad empty heatmap and gt.
        if len(image_v_list) < min_nr_prop:
            for i in range(min_nr_prop - len(image_v_list)):
                image_v_list.append(np.zeros((l_y, l_r), dtype=np.uint8))
                pred_kpts.append((0, 0))
                gt_kpts.append((0, 0))
        vis_2d_heatmap_ver(heatmap_list=image_v_list, filename=save_image_path,
                           gt_point_list=gt_kpts, pred_point_list=pred_kpts,
                           n_max_prop=max(min_nr_prop, int((len(image_v_list) + 1) / 2) * 2))
    del image_v_list


def gen_aligned_pose_heatmap(proposals, heatmap_h, heatmap_v, gt_pose, pred_pose, file_dir, base_time, verbose=True):
    """
    plot and save aligned pose heatmap.
    :param proposals: list (t, n_prop, 4)
    :param heatmap_h: list (t, n_prop, (14, 48, 48) )
    :param heatmap_v: list (t, n_prop, (14, 128, 48) )
    :return:
    """
    ensure_dir(file_dir)
    image_tag = "{kpt_name:}_{h_v:}_{time:05d}.png"

    def yeild_records():
        r_length = len(proposals)
        for t in range(r_length):

            # align gt pose to pred pose
            gt_threshold = 10
            gt_pose_candidates, gt_poses_aligned = gt_pose[t], []
            for pose in pred_pose[t]:
                # for each pred pose, find nearest one.
                dist = np.asarray([pose_dist(pose, p) for p in gt_pose_candidates if np.sum(p) > 0])
                dist_min_idx = np.argmin(dist)

                # handle no gt with threshold.
                if dist[dist_min_idx] > gt_threshold:
                    gt_poses_aligned.append(np.zeros_like(gt_pose_candidates[dist_min_idx]))
                else:
                    gt_poses_aligned.append(np.array(gt_pose_candidates[dist_min_idx]))

            # distribute work for visualization.
            for idx_kpt in range(pose_config.N_KPTS):
                # number of prop
                nr_prop = len(proposals[t])
                # proposals, heatmap, keypoint
                proposals_t, heatmap_h_t, heatmap_v_t = proposals[t], heatmap_h[t], heatmap_v[t]
                kpt_pred_list = [pred_pose[t][i][idx_kpt] for i in range(nr_prop)]
                kpt_gt_list = [gt_poses_aligned[i][idx_kpt] for i in range(nr_prop)]
                yield (t, idx_kpt, proposals_t, heatmap_h_t, heatmap_v_t, kpt_pred_list, kpt_gt_list)

    _ = map_in_pool(
        fn=partial(_gen_pose_heatmap_per_t_kpt,
                   file_dir=file_dir, base_time=base_time, image_tag=image_tag, verbose=verbose),
        data=yeild_records(),
        single_process=False,
        verbose=verbose,
        nr_pro=16
    )

    gc.collect()


def draw_feature_map(feature, filename, figsize=None, grid=None):
    nr_channel = len(feature)
    if figsize is None:
        figsize = (16, 10)
    if grid is None:
        grid = (int(nr_channel / 8), 8)
    plt.set_cmap('jet')
    fig, axeslist = plt.subplots(ncols=grid[0], nrows=grid[1], figsize=figsize, dpi=100)
    for idx_cha in range(nr_channel):
        axeslist.ravel()[idx_cha].imshow(feature[idx_cha])  # , interpolation='bilinear')
        # Turn off tick labels
        axeslist.ravel()[idx_cha].set_yticklabels([])
        axeslist.ravel()[idx_cha].set_xticklabels([])
    plt.tight_layout()  # optional
    plt.savefig("image.png" if filename is None else filename)
    plt.clf()
    plt.close()


def gen_aligned_feature_map_worker(record, image_tag, file_dir, base_time, verbose=False):
    t, proposal_t, hor, ver = record

    if verbose and t % 10 == 0:
        print("t: {}".format(t))

    nr_channel = len(hor)

    # feature map without cropping
    path_hor = os.path.join(file_dir, image_tag.format(h_v="hor", time=base_time + t) + ".png")
    path_ver = os.path.join(file_dir, image_tag.format(h_v="ver", time=base_time + t) + ".png")
    if not os.path.exists(path_hor):
        draw_feature_map(hor, filename=path_hor, figsize=(8, 8), grid=(8, int(nr_channel / 8)))
    if not os.path.exists(path_ver):
        draw_feature_map(ver, filename=path_ver, figsize=(16, 8))

    # feature map with cropping
    mask_h, mask_v = np.zeros_like(hor), np.zeros_like(ver)
    for prop in proposal_t:
        x_min, r_min, x_max, r_max = prop[0:4] / [rpn_config.FEAT_STRIDE_X, rpn_config.FEAT_STRIDE_Z, rpn_config.FEAT_STRIDE_X, rpn_config.FEAT_STRIDE_Z]
        x_min, r_min, x_max, r_max = int(x_min), int(r_min), int(x_max), int(r_max)
        mask_h[:, x_min:x_max, r_min:r_max] = 1
        mask_v[:, :, r_min:r_max] = 1

    # feature map without cropping
    path_hor = os.path.join(file_dir, image_tag.format(h_v="hor_crop", time=base_time + t) + ".png")
    path_ver = os.path.join(file_dir, image_tag.format(h_v="ver_crop", time=base_time + t) + ".png")
    if not os.path.exists(path_hor):
        draw_feature_map(hor * mask_h, filename=path_hor, figsize=(8, 8), grid=(8, int(nr_channel / 8)))
    if not os.path.exists(path_ver):
        draw_feature_map(ver * mask_v, filename=path_ver, figsize=(16, 8))


def gen_aligned_feature_map(proposals, feature_hor, feature_ver, file_dir, base_time, verbose=True):
    ensure_dir(file_dir)
    image_tag = "{h_v:}_{time:05d}"

    feature_hor, feature_ver = np.moveaxis(feature_hor[0], 0, 1), np.moveaxis(feature_ver[0], 0, 1)

    def yield_records():
        for t in range(len(proposals)):
            yield t, proposals[t], feature_hor[t], feature_ver[t]

    _ = map_in_pool(
        fn=partial(gen_aligned_feature_map_worker,
                   file_dir=file_dir, base_time=base_time, image_tag=image_tag, verbose=verbose),
        data=yield_records(),
        single_process=False,
        verbose=verbose,
        nr_pro=16
    )


KPTS_COLOR = np.array([[255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
                       [0, 85, 255], [0, 0, 255], [85, 0, 255], [255, 0, 0]])


def gen_bottom_up_heatmap_worker(record, image_tag, file_dir, base_time, verbose=True):
    t, i_kpt, kpt_name, hor, ver = record
    if verbose and t % 10 == 0:
        print("t: {}, i_kpt: {}".format(t, i_kpt))
    hor_file_name = os.path.join(file_dir, image_tag.format(
            kpt_name=kpt_name.replace(' ', '-').lower(), h_v="hor", time=base_time + t) + ".png")
    ver_file_name = os.path.join(file_dir, image_tag.format(
            kpt_name=kpt_name.replace(' ', '-').lower(), h_v="ver", time=base_time + t) + ".png")
    if os.path.exists(hor_file_name) and os.path.exists(ver_file_name):
        return
    vis_2d_heatmap(heatmap=hor, figsize=(8, 8), filename=hor_file_name)
    vis_2d_heatmap(heatmap=ver, figsize=(8, 4), filename=ver_file_name)


def gen_bottom_up_heatmap(proposals, bu_hor, bu_ver, gt_pose, file_dir, base_time, verbose=True):

    ensure_dir(file_dir)
    image_tag = "{kpt_name:}_{h_v:}_{time:05d}"

    bu_hor, bu_ver = bu_hor[0], bu_ver[0]

    def yield_record():
        for t in tqdm(range(bu_hor.shape[1])):
            for i_kpt, kpt_name in enumerate(pose_config.KPT_NAMES):
                yield t, i_kpt, kpt_name, bu_hor[i_kpt][t], bu_ver[i_kpt][t]

    map_in_pool(
        fn=partial(gen_bottom_up_heatmap_worker,
                   image_tag=image_tag, file_dir=file_dir, base_time=base_time, verbose=True),
        data=yield_record(),
        single_process=False,
        verbose=verbose,
        nr_pro=8
    )


def gen_bottom_up_color_skeleton_worker(record, image_tag, file_dir, color, base_time, verbose=True):
    t, hor_all_kpt, ver_all_kpt, gt_pose_t = record

    if verbose:
        print("t: {}".format(t))

    hor_colored, ver_colored = hor_all_kpt[:, :, :, np.newaxis] * color, ver_all_kpt[:, :, :, np.newaxis] * color
    hor_colored, ver_colored = np.max(hor_colored, axis=0), np.max(ver_colored, axis=0)
    hor_colored, ver_colored = np.asarray(hor_colored, dtype=np.uint8), np.asarray(ver_colored, dtype=np.uint8)

    hor_gt_path = os.path.join(file_dir, image_tag.format(h_v="hor", time=base_time + t, gt="gt") + ".png")
    hor_path = os.path.join(file_dir, image_tag.format(h_v="hor", time=base_time + t, gt="raw") + ".png")
    ver_gt_path = os.path.join(file_dir, image_tag.format(h_v="ver", time=base_time + t, gt="gt") + ".png")
    ver_path = os.path.join(file_dir, image_tag.format(h_v="ver", time=base_time + t, gt="raw") + ".png")
    if os.path.exists(hor_gt_path) and os.path.exists(hor_path) and os.path.exists(ver_gt_path) and os.path.exists(ver_path):
        return

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(hor_colored)
    plt.savefig(hor_path)
    for i, gt_point_list in enumerate(gt_pose_t):
        xs = [p[2] for p in gt_point_list if p[2] > 0 and p[0] > 0]
        ys = [p[0] for p in gt_point_list if p[2] > 0 and p[0] > 0]
        c = 1 - 0.2 * (i % 3)
        plt.scatter(x=xs, y=ys, s=8, c=(c, c, c), marker='x')  # white point gt
    plt.savefig(hor_gt_path)
    plt.clf()
    plt.close()

    plt.figure(figsize=(12, 6), dpi=100)
    plt.imshow(ver_colored)
    plt.savefig(ver_path)
    for i, gt_point_list in enumerate(gt_pose_t):
        xs = [p[2] for p in gt_point_list if p[2] > 0 and p[1] > 0]
        ys = [p[1] for p in gt_point_list if p[2] > 0 and p[1] > 0]
        c = 1 - 0.2 * (i % 3)
        plt.scatter(x=xs, y=ys, s=8, c=(c, c, c), marker='x')  # white point gt
    plt.savefig(ver_gt_path)
    plt.clf()
    plt.close()


def gen_bottom_up_color_skeleton(proposals, bu_hor, bu_ver, gt_pose, file_dir, base_time, verbose=True):

    ensure_dir(file_dir)
    image_tag = "{h_v:}_{time:05d}_{gt:}"

    bu_hor, bu_ver = bu_hor[0], bu_ver[0]

    def _yield_record():
        for t in range(bu_hor.shape[1]):
            yield t, bu_hor[:, t], bu_ver[:, t], gt_pose[t]

    map_in_pool(
        fn=partial(gen_bottom_up_color_skeleton_worker,
                   image_tag=image_tag, file_dir=file_dir, base_time=base_time, verbose=verbose,
                   color=KPTS_COLOR[:, np.newaxis, np.newaxis, :]),
        data=_yield_record(),
        single_process=False,
        verbose=verbose,
        nr_pro=8,
    )

    # save colored image.
    # color = KPTS_COLOR[:, np.newaxis, np.newaxis, :]
    # for t in tqdm(range(bu_hor.shape[0])):
    #     hor_all_kpt, ver_all_kpt, gt_pose_t = bu_hor[:, t], bu_ver[:, t], gt_pose[t]
    #     hor_colored, ver_colored = hor_all_kpt[:, :, :, np.newaxis] * color, ver_all_kpt[:, :, :, np.newaxis] * color
    #     hor_colored, ver_colored = np.max(hor_colored, axis=0), np.max(ver_colored, axis=0)
    #     hor_colored, ver_colored = np.asarray(hor_colored, dtype=np.uint8), np.asarray(ver_colored, dtype=np.uint8)
    #
    #     hor_gt_path = os.path.join(file_dir, image_tag.format(h_v="hor", time=base_time + t, gt="gt") + ".png")
    #     hor_path = os.path.join(file_dir, image_tag.format(h_v="hor", time=base_time + t, gt="") + ".png")
    #
    #     plt.figure(figsize=(8, 8), dpi=100)
    #     plt.imshow(hor_colored)
    #     plt.savefig(hor_path)
    #     for i, gt_point_list in enumerate(gt_pose_t):
    #         xs = [p[2] for p in gt_point_list if np.sum(p) > 0]
    #         ys = [p[1] for p in gt_point_list if np.sum(p) > 0]
    #         c = 1 - 0.2 * (i % 3)
    #         plt.scatter(x=xs, y=ys, s=8, c=(c, c, c), marker='x')  # white point gt
    #     plt.savefig(hor_gt_path)
    #     plt.clf()
    #     plt.close()
    #
    #     ver_gt_path = os.path.join(file_dir, image_tag.format(h_v="ver", time=base_time + t, gt="gt") + ".png")
    #     ver_path = os.path.join(file_dir, image_tag.format(h_v="ver", time=base_time + t, gt="") + ".png")
    #
    #     plt.figure(figsize=(12, 6), dpi=100)
    #     plt.imshow(ver_colored)
    #     plt.savefig(ver_path)
    #     for i, gt_point_list in enumerate(gt_pose_t):
    #         xs = [p[2] for p in gt_point_list if np.sum(p) > 0]
    #         ys = [p[0] for p in gt_point_list if np.sum(p) > 0]
    #         c = 1 - 0.2 * (i % 3)
    #         plt.scatter(x=xs, y=ys, s=8, c=(c, c, c), marker='x')  # white point gt
    #     plt.savefig(ver_gt_path)
    #     plt.clf()
    #     plt.close()

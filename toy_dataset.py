from __future__ import print_function

import os
import pickle
from functools import partial

import cv2
import lmdb
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import config_rf as cfg
import config_rf as config
from helpers.util import ensure_dir

""" Visualization """


def draw_kpt_3d(kpt3d, filename=None):
    assert kpt3d.shape[-1] == 3 and kpt3d.shape[-2] == 14
    fig = plt.figure(figsize=(8.8, 6.6), dpi=100)
    ax = Axes3D(fig)

    ax.set_xlim([config.BOX_X_3D[0], config.BOX_X_3D[1]])
    ax.set_ylim([config.BOX_R_3D[0], config.BOX_R_3D[1]])
    ax.set_zlim([config.BOX_Y_3D[0], config.BOX_Y_3D[1]])

    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('R', fontsize=20)
    ax.set_zlabel('Y', fontsize=20)

    for idx_pose, pose in enumerate(kpt3d):
        xs = [pose[i][0] for i in range(14)]
        ys = [pose[i][1] for i in range(14)]
        zs = [pose[i][2] for i in range(14)]

        # plot points
        ax.scatter(xs=xs, ys=zs, zs=ys, s=10, c='red', marker='o')

        # draw skeleton lines
        for (i, j) in KPT_LINE_IDX:
            ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]], color='red', linestyle='-', linewidth=1)

    # NOTE(LYC):: invert z axis for good visualization.
    ax = plt.gca()
    ax.invert_zaxis()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def draw_kpt_2d_proj(kpt2d, kpt3d_proj, filename=None):
    assert kpt3d_proj.shape[-1] == 2 and kpt2d.shape[-1] == 2, "{}, {}".format(kpt3d_proj.shape, kpt2d.shape)

    _, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 8), dpi=100)

    def _draw(a, k, c):

        a.set_xlim([0, config.CAMERA_WIDTH])
        a.set_ylim([0, config.CAMERA_HEIGHT])

        # NOTE(LYC):: y axis is reversed for visualization.
        a.invert_yaxis()

        for p in k:
            y, x = p[:, 0], p[:, 1]
            # NOTE(LYC):: y, x is replaced in visualization
            a.scatter(x=y, y=x, s=10, c=c, marker='o')
            for (i, j) in KPT_LINE_IDX:
                a.plot([y[i], y[j]], [x[i], x[j]], color=c, linestyle='-', linewidth=1)

    _draw(ax[0], kpt3d_proj, 'b')
    _draw(ax[1], kpt2d, 'r')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.cla()


def project_3d_point(data, m_camera, vis_img_path, kpt_3d_path):

    kpt2d, kpt3d, idx_frame = data

    if idx_frame % 5 == 0:
        print("idx_frame:", idx_frame)

    nr_camera, nr_3d_prop = m_camera.shape[0], kpt3d.shape[0]
    dim_kpt = 14

    # visualize kpt in 3d
    draw_kpt_3d(kpt3d[:, :, :-1], os.path.join(kpt_3d_path, "{:03d}.png".format(idx_frame)))

    # use home coordinate 3d
    homo_coord_3d = np.array(kpt3d).reshape(-1, 4)
    homo_coord_3d[:, 3] = 1

    # proj to 2d camera coord,
    kpt_3d_proj = np.matmul(m_camera[:, np.newaxis, :, :], homo_coord_3d[np.newaxis, :, :, np.newaxis])
    kpt_3d_proj = kpt_3d_proj.reshape(nr_camera, nr_3d_prop, dim_kpt, 3)
    kpt_3d_proj = kpt_3d_proj / kpt_3d_proj[:, :, :, [-1]]
    # print(kpt_3d_proj.shape)

    # visualize No.0 camera kpt.
    for idx_c in range(nr_camera):
        kpt2d_camera_path = os.path.join(vis_img_path, "kpt2d_camera_{}".format(idx_c))
        draw_kpt_2d_proj(kpt2d[idx_c][:, :, :-1], kpt_3d_proj[idx_c][:, :, :-1],
                         os.path.join(kpt2d_camera_path, "{:03d}.png".format(idx_frame)))


def visualize_projection(kpt2d, kpt3d, parameters):
    # create save path
    vis_img_path = "/afs/csail.mit.edu/u/l/liuyingcheng/lyc_storage/rf-shape/muvs_data/vis"
    kpt_3d_path = os.path.join(vis_img_path, "kpt3d")
    if not os.path.exists(vis_img_path):
        os.makedirs(vis_img_path)
    if not os.path.exists(kpt_3d_path):
        os.makedirs(kpt_3d_path)
    nr_camera = parameters.C.shape[0]
    for idx_c in range(nr_camera):
        kpt2d_camera_path = os.path.join(vis_img_path, "kpt2d_camera_{}".format(idx_c))
        if not os.path.exists(kpt2d_camera_path):
            os.makedirs(kpt2d_camera_path)

    assert len(kpt2d) == len(kpt3d)
    nr_visualize = 100
    assert len(kpt2d) >= nr_visualize

    def _distribute():
        for idx_frame in range(nr_visualize):
            yield kpt2d[idx_frame], kpt3d[idx_frame], idx_frame

    _ = map_in_pool(
        fn=partial(project_3d_point,
                   m_camera=parameters.C, vis_img_path=vis_img_path, kpt_3d_path=kpt_3d_path),
        data=_distribute(),
        single_process=False,
        verbose=True,
        nr_pro=16
    )


def vis_img_proj(kpt2d, kpt3d, img, parameters, vis_img_path=None):
    """
        render and save kpt projection to raw image
    :param kpt2d:
    :param kpt3d:
    :param parameters: camera_parameter
    :param vis_img_path: root to save vis images
    :return:
    """

    assert len(kpt2d) == len(kpt3d)

    if vis_img_path is None:
        vis_img_path = ensure_dir("/afs/csail.mit.edu/u/l/liuyingcheng/lyc_storage/rf-shape/muvs_data/vis_img_proj")
    kpt_3d_path = ensure_dir(os.path.join(vis_img_path, "kpt3d"))
    nr_camera = parameters.C.shape[0]
    for idx_c in range(nr_camera):
        kpt2d_camera_path = ensure_dir(os.path.join(vis_img_path, "kpt2d_camera_{}".format(idx_c)))

    nr_visualize = 100
    assert len(kpt2d) >= nr_visualize

    for idx_vis in range(nr_visualize):
        k3, k2, i = kpt3d[idx_vis], kpt2d[idx_vis], img[idx_vis]


""" Data """


class CameraParameters:
    """
        Parse the camera parameter from hdf5 files
        C: global to camera coordinates
        R: rotation
        T: translation
        I: intrinsic

        C = I * [R, T]
        C = np.dot(I, np.hstack((R, T)))
    """

    def __init__(self, params_folder, camera_list):
        self.params_folder = params_folder
        self.camera_list = camera_list
        self.R, self.T, self.I, self.C = self.load_parameters()

    def __str__(self):
        return "R {} \n T {} \n I {} \n C {}" \
            .format(self.R, self.T, self.I, self.C)

    def load_parameters(self):
        R, T, I, C = [], [], [], []
        for camera in self.camera_list:
            file_name = os.path.join(self.params_folder, "{}.h5".format(camera))
            f = h5py.File(file_name, "r")
            R.append(np.array(f['R']))
            T.append(np.array(f['T']))
            I.append(np.array(f['I']))
            C.append(np.array(f['C']))
        return np.array(R), np.array(T), np.array(I), np.array(C)


def is_exp_folder(folder):
    """ Test if a folder records an experiment """
    location, exp = os.path.split(folder)
    # folder name format check
    parts = location.split('-')
    if not len(parts) == 3:
        return False
    if not ("exp" in parts[0] and len(parts[0]) == 7 and len(parts[1]) == 1 and len(parts[2]) == 3):
        return False
    if not len(exp) == 3:
        return False
    return True


class RFDataset:
    def __init__(self, name, train_exps, val_exps, raw_rf_fps=45, raw_camera_fps=30, other_exps=[]):
        assert not list(set(train_exps).intersection(val_exps))
        self.name = name
        self.train_exps = train_exps
        self.val_exps = val_exps
        self.other_exps = other_exps
        if len(train_exps) >= 2:
            self.visualize_exps = val_exps + train_exps[0:2]
        else:
            self.visualize_exps = val_exps
        self.all_exps = sorted(train_exps + val_exps + other_exps)
        self.raw_rf_fps = raw_rf_fps
        self.raw_camera_fps = raw_camera_fps
        for exp in self.all_exps:
            assert is_exp_folder(exp)


TOY_3D_DATASET = RFDataset(
    name="toy",
    train_exps=["exp1001-2-002/005"],  # , "exp1001-2-002/006"],
    val_exps=[],
    other_exps=[],
    raw_rf_fps=45,
    raw_camera_fps=30
)


def get_lmdb_files(folder):
    if not os.path.exists(folder):
        return 0

    env = lmdb.open(folder, readonly=True)
    with env.begin() as txn:
        num_files = int(txn.get("num-files".encode("utf-8")).decode("utf-8"))
    env.close()
    return num_files


def gather_data(num_files, cams, kpt_folder):
    envs = [lmdb.open(os.path.join(kpt_folder, camera), readonly=True) for camera in tqdm(cams)]
    txns = [env.begin() for env in tqdm(envs)]
    all_data = [load_frame(idx, txns) for idx in tqdm(range(num_files))]
    return all_data


def load_frame(idx, txns):
    """ frame-level data loader across cameras"""
    kpts = []
    for txn in txns:
        filename = "%.6d" % idx
        data = txn.get(filename.encode("utf-8"))
        try:
            camera_kpt = pickle.loads(data)
        except:
            filename = "%d" % idx
            data = txn.get(filename.encode("utf-8"))
            camera_kpt = pickle.loads(data)

        kpts.append(person_filter(camera_kpt))
    return kpts


def person_filter(camera_kpt):
    if len(camera_kpt) == 0:
        return np.zeros((0, cfg.C, 3))
    else:
        res = []
        for i, person in enumerate(camera_kpt):
            idx = np.where(person[:, 2] > 1e-8)[0]
            if len(idx) >= cfg.MIN_DETECTED_POINTS and person[idx, 2].mean() > cfg.CONFIDENCE_3D_THREASHOLD:
                center_w = np.mean(person[idx, 0])
                center_h = np.mean(person[idx, 1])
                if center_w < cfg.BNDRY * cfg.CAMERA_WIDTH or center_w > (1-cfg.BNDRY) * cfg.CAMERA_WIDTH:
                    continue
                if center_h < cfg.BNDRY * cfg.CAMERA_HEIGHT or center_h > (1-cfg.BNDRY) * cfg.CAMERA_HEIGHT:
                    continue
                res.append(person)
        if len(res) == 0:
            return np.zeros((0, cfg.C, 3))
        else:
            return np.array(res)


def read_2d_key_point(exp, loc):
    """
        read 2d key point from data: loc, exp
    :param exp: experiment number
    :param loc: experiment location
    :return: list (len time), list (nr_camera), np array (nr_prop, 14, 3)
    """
    # path
    kpt_folder = config.OPENPOSE_KPT_EXP_FOLDER_LMDB.format(fps=config.CAMERA_FPS, loc=loc, exp=exp)
    cams = sorted(os.listdir(kpt_folder))
    kpt_2d_folders = [config.OPENPOSE_KPT_FOLDER_LMDB.format(fps=config.CAMERA_FPS, loc=loc, exp=exp, cam=cam)
                      for cam in cams]
    skeleton_3d_folder = config.SKELETON_3D_FOLDER.format(fps=config.CAMERA_FPS, loc=loc, exp=exp)
    skeleton_3d_file = config.SKELETON_3D_RAW_FILE.format(fps=config.CAMERA_FPS, loc=loc, exp=exp)
    print("kpt_folder:", kpt_folder)
    print("cam:", cams)
    print("kpt_2d_folders:", kpt_2d_folders)

    print("checking kpt 2d folders")
    bad = False
    for folder in kpt_2d_folders:
        if not os.path.exists(folder):
            print("Folder %s does not exist" % folder)
            bad = True
            continue

        try:
            env = lmdb.open(folder, readonly=True)
        except lmdb.Error:
            print('Database %s does not exist' % folder)
            bad = True
            continue
        with env.begin() as txn:
            if txn.get("done".encode("utf-8")) is None:
                print("Folder %s is not fully ready" % folder)
                bad = True
    if bad:
        return -1
    print("Finished all kpt 2d files.")

    rf_ds = RFDataset(
        name="rf_ds",
        train_exps=["{}/{}".format(loc, exp)],
        val_exps=[],
        other_exps=[],
        raw_rf_fps=45,
        raw_camera_fps=30
    )

    len_kpt_2d_folders = [get_lmdb_files(folder) for folder in kpt_2d_folders]

    print('3d skeleton is written to file {}'.format(skeleton_3d_file))

    num_files = len_kpt_2d_folders[0]
    all_data = gather_data(num_files, cams, kpt_folder)

    return all_data


def read_3d_key_point(exp, loc):
    """
        read 3d key point from data: loc, exp
    :param exp: experiment number, 'exp0606-2-001'
    :param loc: experiment location, '006'
    :return: list (len time), np array (nr_instance, 14, 4)
    """

    pose_folder = config.SKELETON_3D_SMOOTHED_LMDB.format(fps=cfg.CAMERA_FPS, loc=loc, exp=exp)
    print("pose_folder:", pose_folder)

    pose_env = lmdb.open(pose_folder, readonly=True, lock=False, readahead=False)

    with pose_env.begin(write=False) as txn:
        num_files = int(txn.get("num-files".encode("utf-8")).decode("utf-8"))
        all_2d_kpt = []
        for i in tqdm(range(num_files)):
            data = txn.get(str(i).encode("utf-8"))
            poses = pickle.loads(data)
            poses = filter_poses(poses)
            all_2d_kpt.append(poses)

    return all_2d_kpt


def filter_poses(poses):
    new_poses = []

    for idx, pose in enumerate(poses):
        new_pose = np.zeros((cfg.C, 4))
        pose = np.array(pose['keypoints'])

        for k_id, kpt in enumerate(pose):
            if cfg.BOX_X_3D[0] < kpt[0] < cfg.BOX_X_3D[1] and cfg.BOX_Y_3D[0] < kpt[1] < cfg.BOX_Y_3D[1] \
                    and cfg.BOX_R_3D[0] < kpt[2] < cfg.BOX_R_3D[1]:
                    new_pose[k_id] = kpt

        good_idx = np.where(new_pose[:, 3] > 1e-8)[0]
        if len(good_idx) >= cfg.MIN_DETECTED_POINTS and new_pose[good_idx, 3].mean() > cfg.CONFIDENCE_3D_THREASHOLD:
            new_poses.append(new_pose)

    return np.asarray(new_poses)


def parse_trajectory(kpt3d, m_camera):
    """ associate proposals based on distance. """

    def _dist(a, b):
        assert a.shape == (14, 4)
        assert b.shape == (14, 4)
        valid_kpt = np.logical_and(a[:, -1] > 0.9, b[:, -1] > 0.9)
        if np.sum(valid_kpt) == 0:
            return 1e10
        d = np.sqrt(np.square(a[:, :-1] - b[:, :-1]).sum(axis=1))
        return np.sum(d * valid_kpt) / valid_kpt.sum()

    nr_traj = len(kpt3d[0])  # track proposals appear in first frame.
    history = [[] for _ in range(nr_traj)]

    # tracking
    for i_prop in range(nr_traj):
        history[i_prop].append(i_prop)
    for t in range(1, len(kpt3d), 1):
        print("t: {}".format(t))
        for idx_prop in range(nr_traj):
            prev_kpt = kpt3d[t - 1][history[idx_prop][t - 1]]
            candidates = kpt3d[t]
            d = [_dist(prev_kpt, k) for k in kpt3d[t]]
            print("idx_prop = {}, d = {}".format(idx_prop, d))
            history[idx_prop].append(np.argmin(d))

    # check: visualize trajectory
    def _check_vis():
        save_root = "./vis_traj_3d"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for idx_traj in range(nr_traj):
            print("idx_traj: {}".format(idx_traj))
            kpt_traj = [kpt3d[t][i][:, :-1] for t, i in enumerate(history[idx_traj])]
            traj_path = os.path.join(save_root, "troj_{:02d}".format(idx_traj))
            if not os.path.exists(traj_path):
                os.mkdir(traj_path)
            for i, k in enumerate(tqdm(kpt_traj)):
                filename = os.path.join(traj_path, "{:03d}.png".format(i))
                draw_kpt_3d(k.reshape(1, *k.shape), filename)

    # save trajectory and camera parameter
    toy_data_root = "data_toy"
    if not os.path.exists(toy_data_root):
        os.mkdir(toy_data_root)
    traj = []
    for idx_traj in range(nr_traj):
        traj.append([kpt3d[t][i][:, :-1] for t, i in enumerate(history[idx_traj])])
    np.save(os.path.join(toy_data_root, "trajs.npy"), np.asarray(traj))
    np.save(os.path.join(toy_data_root, "m_camera.npy"), np.asarray(m_camera))


def read_image(loc, exp, nr_img=None):
    """
        load image
    :param loc: experiment location, 'exp0606-2-001'
    :param exp: experiment number, '006'
    :param nr_img: number of images to load
    :return: list (nr_camera, nr_img, *shape_img)
    """

    sync_camera_exp_folder = cfg.SYNC_CAMERA_EXP_FOLDER_LMDB.format(fps=config.CAMERA_FPS, loc=loc, exp=exp)
    print("Loading camera images from:", sync_camera_exp_folder)
    if not os.path.exists(sync_camera_exp_folder):
        raise ValueError("Synchronized folder {} doesn't exist.".format(sync_camera_exp_folder))
    cameras = sorted(os.listdir(sync_camera_exp_folder))

    images = []
    for cam in cameras:
        print("cam: ", cam)
        img_folder_lmdb = config.SYNC_CAMERA_FOLDER_LMDB.format(fps=config.CAMERA_FPS, loc=loc, exp=exp, cam=cam)
        env = lmdb.open(img_folder_lmdb, readonly=True)
        this_images = []
        with env.begin() as txn:
            assert txn.get("done".encode("utf-8")) is not None, "Images file is not fully done"
            entries = int(txn.get("num-files".encode("utf-8")).decode("utf-8"))
            if nr_img is not None:
                entries = min(entries, nr_img)
            for idx in tqdm(range(entries)):
                img = txn.get(str(idx).encode("utf-8"))
                nparr = np.fromstring(img, np.uint8)
                img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                this_images.append(img_np)
        images.append(this_images)
    return images


""" main """


def main():

    # toy exp
    # loc, exp = "exp1001-2-002", "005"
    loc, exp = "exp0606-2-001", "006"
    print("use toy exp {}, loc {}".format(exp, loc))

    # kpt2d = read_2d_key_point(exp, loc)
    # kpt3d = read_3d_key_point(exp, loc)
    read_image(loc, exp)

    from IPython import embed
    embed(header="after kpt loading ")

    # kpt2d: list (len time), list (nr_camera), np array (nr_prop, 14, 3)
    # kpt3d: list (len time), np array (3, 14, 4)
    # m_camera: (10, 3, 4)

    params_folder = config.CAMERA_PARAMS_FOLDER.format(loc=loc)
    kpt_folder = config.OPENPOSE_KPT_EXP_FOLDER_LMDB.format(fps=config.CAMERA_FPS, loc=loc, exp=exp)
    cams = sorted(os.listdir(kpt_folder))
    parameters = CameraParameters(params_folder, cams)

    # visualize 3d to 2d camera space projection and compare with 2d raw kpt.
    visualize_projection(kpt2d, kpt3d, parameters)

    # parse and save trajectory
    # nr_sample_frame = 50
    # parse_trajectory(kpt3d[:nr_sample_frame], parameters.C)


if __name__ == "__main__":
    main()

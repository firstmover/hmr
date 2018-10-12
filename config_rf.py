import socket
import os
import numpy as np
from os.path import join

# Global parameters
LEN_SAMPLE = 50
NONOVERLAP = 25
N_KPTS = C = 14
CAMERA_HEIGHT = CAM_H = 1232
CAMERA_WIDTH = CAM_W = 1640

# The net_resolution parameter of OpenPose. It should be multiples of 16
OP_NET_W = 448
OP_NET_H = 336
OP_POSE_TYPE = "COCO"

NUM_CAMERAS = 8

DLT_MAX_ITERS = 10
DLT_MAX_ERROR_RATIO = 10

HEAD_HEATMAP_INDICES = [0, 14, 15, 16, 17]
MASK_RADIUS = 0.0
N_DIGITS = 6
CONFIDENCE_THREASHOLD = 0.6
CONFIDENCE_3D_THREASHOLD = 0.5
LINE_DIST_THRESHOLD = 250
PROJ_DIST_THRESHOLD = 100
MIN_DETECTED_POINTS = 5
MIN_ASSOCIATE_POINTS = 2
BNDRY = 0.03
MAX_NUM_BBOX = 10

CAMERA_FPS = 15
RF_FPS = 30

CAMERA_MIN_VIEWS = 2

BOX_MARGIN = 0
BOX_X_3D = [-5000 - BOX_MARGIN, 5000 + BOX_MARGIN]
BOX_Y_3D = [-1500 - BOX_MARGIN, 1500 + BOX_MARGIN]
BOX_R_3D = [1000 - BOX_MARGIN, 11000 + BOX_MARGIN]
BBOX_EXTEND = 200

HOR_SHAPE = (2, 200, 200)
VER_SHAPE = (2, 60, 200)
INPUT_SHAPE = (HOR_SHAPE[1], VER_SHAPE[1], HOR_SHAPE[2])
RF_HOR_BYTES = np.prod(HOR_SHAPE) * 4
RF_VER_BYTES = np.prod(VER_SHAPE) * 4
RF_FRAME_BYTES = RF_HOR_BYTES + RF_VER_BYTES

HEATMAP_X = np.linspace(-5, 5, 200)
HEATMAP_Y = np.linspace(-1.5, 1.5, 60)
HEATMAP_Z = np.linspace(1, 11, 200)
assert len(HEATMAP_X) == HOR_SHAPE[1]
assert len(HEATMAP_Y) == VER_SHAPE[1]
assert len(HEATMAP_Z) == HOR_SHAPE[2] and len(HEATMAP_Z) == VER_SHAPE[2]
KPT_INDS2NAME = ["Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
                 "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Head"]

KPT_NAME2INDS = {"Neck": 0, "RShoulder": 1, "RElbow": 2, "RWrist": 3, "LShoulder": 4, "LElbow": 5, "LWrist": 6,
                 "RHip": 7, "RKnee": 8, "RAnkle": 9, "LHip": 10, "LKnee": 11, "LAnkle": 12, "Head": 13}

# space output size of feature net.
BU_HEATMAP_HOR_SHAPE, BU_HEATMAP_VER_SHAPE = (200, 200), (60, 200)
BU_HEATMAP_SHAPE = (BU_HEATMAP_HOR_SHAPE[0], BU_HEATMAP_VER_SHAPE[0], BU_HEATMAP_HOR_SHAPE[1])
BU_DOWN_SAMPLE_RATE = [i / float(j) for i, j in zip(INPUT_SHAPE, BU_HEATMAP_SHAPE)]

# Machine-dependent paths
name = socket.gethostname()

# Non-linear coefficients of the antenna array
NON_LINEAR_COEFFICIENTS = "/data/netmit/rf-vision/3d/nonlinear_array_coef/nonlinear_coefficients.npz"
NON_LINEAR_COEFFICIENTS_XYR = "/data/netmit/rf-vision/3d/nonlinear_array_coef/nonlinear_coefficients_xyr.npz"

RAW_DATA_PATH = "/data/netmit/rf-vision/3d/raw"
DATA_PATH = "/data/netmit/rf-vision/3d/processed"
LIB_OPENPOSE_PATH = "/usr/local/openpose"
VIZ_PATH = "/data/netmit/rf-vision/3d/processed/viz"
CODE_BACKUP_PATH = "/afs/csail.mit.edu/u/m/mingmin/backup/rf-vision/3d"

IMAGES_DATA_PATH = "/data/scratch-oc40/rf-vision/3d/processed"

# Remote (laptop) data folders
REMOTE_RAW_CALIBRATE_FOLDER = join("{log_path}", "{loc}", "calibration")
REMOTE_RAW_CAMERA_FOLDER = join("{log_path}", "{loc}", "{exp}", )
REMOTE_RAW_RF_FOLDER = join("{log_path}", "{loc}", "{exp}")
REMOTE_RAW_CALIBRATE_PARAMS = join("{log_path}", "{loc}", "camera_matrices")

# Raw data folders
RAW_CALIBRATE_TAR_FILE = join(RAW_DATA_PATH, "calibration", "{loc}.tar.gz")
RAW_CAMERA_LOCATION_FOLDER = join(RAW_DATA_PATH, "camera", "{loc}")
RAW_CAMERA_EXP_FOLDER = join(RAW_DATA_PATH, "camera", "{loc}", "{exp}")
RAW_CAMERA_FOLDER = join(RAW_DATA_PATH, "camera", "{loc}", "{exp}", "{cam}")
RAW_RF_FOLDER = join(RAW_DATA_PATH, "rf", "{loc}", "{exp}")
RAW_RF_TEMP_FOLDER = join("/tmp", "rf", "{loc}", "{exp}")

SYNC_CALIBRATE_FOLDER = join(DATA_PATH, "sync", "calibration", "{loc}")
SYNC_CAMERA_EXP_FOLDER = join(IMAGES_DATA_PATH, "sync", "camera", "fps-{fps}", "{loc}", "{exp}")
SYNC_CAMERA_EXP_FOLDER_LMDB = join(IMAGES_DATA_PATH, "sync", "camera", "fps-{fps}", "{loc}", "{exp}-lmdb")
SYNC_CAMERA_FOLDER = join(IMAGES_DATA_PATH, "sync", "camera", "fps-{fps}", "{loc}", "{exp}", "{cam}")
SYNC_CAMERA_FOLDER_LMDB = join(IMAGES_DATA_PATH, "sync", "camera", "fps-{fps}", "{loc}", "{exp}-lmdb", "{cam}")
SYNC_CAMERA_FOLDER_TMP = join("/tmp", "sync", "camera", "fps-{fps}", "{loc}", "{exp}", "{cam}")
SYNC_RF_FOLDER_BIN = join(DATA_PATH, "sync", "rf", "{array_type}", "fps-{fps}", "{loc}", "{exp}-bin")
SYNC_RF_SUB_MEDIAN_FOLDER = join(DATA_PATH, "sync", "rf-sub-median", "{array_type}", "fps-{fps}", "{loc}", "{exp}")
SYNC_RF_MEDIAN_NPZ = join(DATA_PATH, "sync", "rf-median", "{array_type}", "fps-{fps}", "{loc}-{exp}.npz")

OPENPOSE_HTMP_EXP_FOLDER = join(DATA_PATH, "openpose", "heatmap", "fps-{fps}", "{loc}", "{exp}")
OPENPOSE_LOCAL_HTMP_FOLDER = join("/tmp", "openpose", "heatmap", "fps-{fps}", "{loc}", "{exp}", "{cam}")
OPENPOSE_KPT_EXP_FOLDER_LMDB = join(DATA_PATH, "openpose", "keypoint", "fps-{fps}", "{loc}", "{exp}-lmdb")
OPENPOSE_KPT_FOLDER_LMDB = join(DATA_PATH, "openpose", "keypoint", "fps-{fps}", "{loc}", "{exp}-lmdb", "{cam}")
OPENPOSE_LOCAL_KPT_FOLDER = join("/tmp", "openpose", "keypoint", "fps-{fps}", "{loc}", "{exp}", "{cam}")
OPENPOSE_LOCAL_KPT_JSON_FOLDER = join("/tmp", "openpose", "keypoint-json", "fps-{fps}", "{loc}", "{exp}", "{cam}")

CAMERA_PARAMS_FOLDER = join(DATA_PATH, "camera-parameters", "{loc}")
SKELETON_3D_FOLDER = join(DATA_PATH, "skeleton3d", "fps-{fps}", "{loc}", "{exp}")
SKELETON_3D_RAW_FILE = join(DATA_PATH, "skeleton3d", "fps-{fps}", "{loc}", "{exp}", "skeleton_raw.json")
SKELETON_3D_SMOOTHED_FILE = join(DATA_PATH, "skeleton3d", "fps-{fps}", "{loc}", "{exp}", "skeleton_smoothed.json")
SKELETON_3D_SMOOTHED_LMDB = join(DATA_PATH, "skeleton3d", "fps-{fps}", "{loc}", "{exp}", "skeleton-smoothed-lmdb")
VIZ_FOLDER = join(VIZ_PATH, "views-{views}", "fps-{fps}", "{loc}", "{exp}")

RAW_VIDEO_FOLDER = join(DATA_PATH, "raw-video", "raw", "{loc}", "{exp}")
CAMERA_SYSTEM_VIDEO_FOLDER = join(DATA_PATH, "raw-video", "cameras", "{loc}", "{exp}")

# Trained models
MODEL_PATH = "/data/netmit/rf-vision/3d/models"

CHECKPOINT_PATH = join("/data/netmit/rf-vision/3d/models", "{model_name}", "checkpoint_{epoch}.pth.tar")

# Prediction Folder Structures
PREDICTION_PATH = "/data/netmit/rf-vision/3d/predictions"

PRED_MODEL_PATH = join(PREDICTION_PATH, "{model_name}", "{epoch}")
PREDICTION_FOLDER = join(PREDICTION_PATH, "{model_name}", "{epoch}", "{loc}", "{exp}")

PRED_SKELETON_3D_FILE = join(PREDICTION_FOLDER, "skeleton_raw.json")
PRED_SKELETON_3D_SMOOTHED_FILE = join(PREDICTION_FOLDER, "skeleton_smoothed.json")

PRED_RCNN_BBOX_FILE = join(PREDICTION_FOLDER, "rcnn_bbox.json")
PRED_RPN_BBOX_FILE = join(PREDICTION_FOLDER, "rpn_bbox.json")

# Dataset configuration
DATA_CONFIG_PATH = "/data/netmit/rf-vision/3d/config"
DATASET_PATH = join(DATA_CONFIG_PATH, "datasets")
DATASET_LEN_FILE = join(DATA_CONFIG_PATH, "dataset_len.json")

""" keypoint """

KPT_NAMES = [
    "neck",
    "R shoulder",
    "R elbow",
    "R wrist",
    "L shoulder",
    "L elbow",
    "L wrist",
    "R hip",
    "R knee",
    "R ankle",
    "L hip",
    "L knee",
    "L ankle",
    "head"]

KPT_LINE = [
    ("head", "neck"),
    ("neck", "L shoulder"),
    ("neck", "R shoulder"),
    ("L shoulder", "L elbow"),
    ("R shoulder", "R elbow"),
    ("L elbow", "L wrist"),
    ("R elbow", "R wrist"),
    ("neck", "L hip"),
    ("neck", "R hip"),
    ("L hip", "R hip"),
    ("L hip", "L knee"),
    ("R hip", "R knee"),
    ("R knee", "R ankle"),
    ("L knee", "L ankle")
]

KPT_LINE_IDX = [(KPT_NAMES.index(t[0]), KPT_NAMES.index(t[1])) for t in KPT_LINE]

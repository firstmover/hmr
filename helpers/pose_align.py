import numpy as np
import torch

from models.utils import CoordinatesTransformer
from models.utils import xyr2xyz
import models.RPN.config as rpn_cfg


coordinator = CoordinatesTransformer()


def pose_align(gt_poses, proposals):
    x_center = torch.unsqueeze(proposals[:, 2] + proposals[:, 0], dim=1) / 2
    y_center = (rpn_cfg.INPUT_Y + 1) / 2
    z_center = torch.unsqueeze(proposals[:, 3] + proposals[:, 1], dim=1) / 2
    gt_poses[:, :, 0] = gt_poses[:, :, 0] - x_center
    gt_poses[:, :, 1] = gt_poses[:, :, 1] - y_center
    gt_poses[:, :, 2] = gt_poses[:, :, 2] - z_center
    return gt_poses


def inv_pose_align(aligned_pose, proposals):
    pose = aligned_pose[:, :, 0:3].clone()
    x_center = torch.unsqueeze(proposals[:, 2] + proposals[:, 0], dim=1) / 2
    y_center = (rpn_cfg.INPUT_Y + 1) / 2
    z_center = torch.unsqueeze(proposals[:, 3] + proposals[:, 1], dim=1) / 2
    pose[:, :, 0] = aligned_pose[:, :, 0] + x_center
    pose[:, :, 1] = aligned_pose[:, :, 1] + y_center
    pose[:, :, 2] = aligned_pose[:, :, 2] + z_center
    return pose


def pose_aligned2world(aligned_pose, proposals):
    htmp_pose = inv_pose_align(aligned_pose, proposals)
    world_pose = coordinator.htmp2world(htmp_pose)
    return world_pose

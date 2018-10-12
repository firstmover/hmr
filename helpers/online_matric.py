import numpy as np
from models.RCNN import config as pose_config


def pose_dist(p1, p2):
    s = []
    for i in range(pose_config.N_KPTS):
        if np.sum(p1[i]) > 0.1 and np.sum(p2[i]) > 0.1:
            s.append(np.sqrt(np.sum(np.square(p1[i] - p2[i]))))
    return np.mean(s)


# oks metric, sigma
sigmas = np.array([.79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .79]) / 10.0 * np.sqrt(3.0 / 2.0)
bbox_scale = 1000  # NOTE(LYC):: normalize key point

#     "neck",       .79
#     "R shoulder", .79
#     "R elbow",    .72
#     "R wrist",    .62
#     "L shoulder", .79
#     "L elbow",    .72
#     "L wrist",    .62
#     "R hip",      1.07
#     "R knee",     .87
#     "R ankle",    .89
#     "L hip",      1.07
#     "L knee",     .87
#     "L ankle",    .89
#     "head"        .79


def pose_oks(gt_pose, pred_pose):
    # get oks distance between gt and pred pose
    assert gt_pose.shape[0] == 14 and pred_pose.shape[0] == 14
    assert gt_pose.shape[1] == 3 and pred_pose.shape[1] == 3, "{} {}".format(gt_pose, pred_pose)
    dist_square = (((gt_pose - pred_pose) / bbox_scale) ** 2).sum(axis=1)
    visible = (gt_pose.sum(axis=1) > 0.1)
    return (np.exp(-dist_square / (2 * sigmas ** 2)) * visible).sum() / (visible.sum() + 1e-10)


class RPNMeter:

    def __init__(self):
        self.nr_pred = 0
        self.nr_gt = 0
        self.connected_pred = 0
        self.connected_gt = 0
        self.connections = 0
        self.pose_dist_threshold = 500  # TODO(LYC):: adjust this threshold

    def add(self, pred, gt, report=True):

        connected_pred, connected_gt, connections = 0, 0, 0
        sum_gt, sum_pred = 0, 0

        for t, (p_pose, g_pose) in enumerate(zip(pred, gt)):
            nr_pred, nr_gt = len(p_pose), len(g_pose)

            edge = np.zeros((nr_gt, nr_pred), dtype=np.bool)
            for i in range(nr_gt):
                for j in range(nr_pred):
                    edge[i][j] += (pose_dist(p_pose[j][:, :3], g_pose[i][:, :3]) < self.pose_dist_threshold)

            connected_pred += (edge.sum(axis=0) > 0).sum()
            connected_gt += (edge.sum(axis=1) > 0).sum()
            connections += edge.sum()
            sum_pred += nr_pred
            sum_gt += nr_gt

        self.connected_pred += connected_pred
        self.connected_gt += connected_gt
        self.connections += connections
        self.nr_pred += sum_pred
        self.nr_gt += sum_gt

        if report:
            print("perframe: p {:.4f}, r {:.4f}, d {:.4f}".format(
                connected_pred / sum_pred, connected_gt / sum_gt, connections / connected_gt
            ))
            print("all: p {:.4f}, r {:.4f}, d {:.4f}".format(
                self.connected_pred / self.nr_pred, self.connected_gt / self.nr_gt, self.connections / self.connected_gt
            ))


class APMeter:
    def __init__(self, oks_threshold):
        self.oks_threshold = oks_threshold

        self.tags = []
        self.confs = []
        self.total_gt = 0

    def add(self, gt_poses, pred_poses, pred_conf):
        # number of positives
        npos = sum([len(gt) for gt in gt_poses])
        self.total_gt += npos

        # get oks distance of pred and ground truth for each frame
        poses, tags, confs = match_skeletons(gt_poses, pred_poses, pred_conf, self.oks_threshold)

        self.tags.extend(tags)
        self.confs.extend(confs)

    def compute_ap(self, print_info):
        tags = np.concatenate(self.tags)
        confs = np.concatenate(self.confs)
        sorted_inds = np.argsort(-confs)
        tags = tags[sorted_inds]

        # compute the precision recall curve
        tp = tags.copy()
        fp = (1 - tags).copy()

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(self.total_gt)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)

        s_rec, s_prec, ap = voc_ap(rec, prec)

        if print_info:
            print("AP @ {:.2f}: {:.6f}".format(self.oks_threshold, ap))
        return ap


class MAPMeter:
    def __init__(self, oks_thresholds):
        self.oks_thresholds = oks_thresholds
        self.ap_meters = [APMeter(oks) for oks in oks_thresholds]

    def add(self, gt_poses, pred_poses, pred_conf):
        for ap_meter in self.ap_meters:
            ap_meter.add(gt_poses, pred_poses, pred_conf)

    def compute_ap(self, print_info):
        all_AP = [ap_meter.compute_ap(print_info) for ap_meter in self.ap_meters]
        mAP = np.array(all_AP).mean()
        if print_info:
            print("mAP: {:.8f}".format(mAP))
        return mAP


def match_skeletons(gt_pose_all, pred_pose_all, pred_conf_all, oks_threshold):
    assert len(gt_pose_all) == len(pred_pose_all) and len(pred_pose_all) == len(pred_conf_all)

    tags_all, poses_all, confs_all = [], [], []

    for gt, pose, conf in zip(gt_pose_all, pred_pose_all, pred_conf_all):
        gt, pose, conf = np.array(gt), np.array(pose), np.array(conf)

        if len(gt) == 0:
            tags_all.append(np.zeros(len(pose)))
            poses_all.append(pose)
            continue

        if len(pose) == 0:
            tags_all.append(np.zeros(0))
            poses_all.append(pose)
            continue

        # sort by confidence
        sorted_ind = np.argsort(-conf)
        pose = pose[sorted_ind, :, :]
        conf = conf[sorted_ind]

        # get oks distance between gt and pred
        oks_dist = np.asarray([[pose_oks(p[:, :3], g[:, :3]) for g in gt] for p in pose])

        # find max for each pred
        oks_max, oks_max_idx = oks_dist.max(axis=1), oks_dist.argmax(axis=1)

        # get true positives
        det_tag, matched_gt = [], [False] * len(gt)
        for i in range(len(pose)):
            if oks_max[i] >= oks_threshold and not matched_gt[oks_max_idx[i]]:
                det_tag.append(1)
                matched_gt[oks_max_idx[i]] = True
            else:
                det_tag.append(0)

        tags_all.append(np.array(det_tag))  # contain True for true positive
        poses_all.append(pose)
        confs_all.append(conf)

    return poses_all, tags_all, confs_all


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return mrec[:-1], mpre[:-1], ap

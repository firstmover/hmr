import os
import cv2
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from models.RCNN import config as pose_config


class OnlinePoseDisplayer:

    def __init__(self):
        cv2.startWindowThread()
        # NOTE(LYC):: very weird, can't use self.window_name and pass it..
        cv2.namedWindow("pose_3d")

        self.fig = plt.figure(figsize=(8.8, 6.6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = Axes3D(self.fig)

    def disp_pose_3d(self, pred_pose, gt_pose=None):

        assert len(pred_pose) == len(gt_pose)

        time_range = len(pred_pose)

        fig, canvas, ax = self.fig, self.canvas, self.ax

        # iter frames
        for t in range(time_range):

            ax.set_xlim([-5000, 5000])
            ax.set_zlim([-1500, 1500])
            ax.set_ylim([1000, 11000])

            ax.set_xlabel('X', fontsize=20)
            ax.set_ylabel('R', fontsize=20)
            ax.set_zlabel('Y', fontsize=20)

            ax.view_init(elev=30)

            # prediction
            for idx_pose, pose in enumerate(pred_pose[t]):

                xs = [pose["keypoints"][i][0] for i in range(14)]
                ys = [-pose["keypoints"][i][1] for i in range(14)]
                zs = [pose["keypoints"][i][2] for i in range(14)]

                # put vertical axis y to third axis of matplotlib
                ax.scatter(xs, zs, ys, s=10, c='red', marker='o')

                # draw skeleton lines
                for (i, j) in pose_config.KPT_LINE_IDX:
                    ax.plot([xs[i], xs[j]], [zs[i], zs[j]], [ys[i], ys[j]], color='red', linestyle='-', linewidth=1)

            # ground truth
            if gt_pose is not None:
                for idx_pose, pose in enumerate(gt_pose[t]):
                    # skip gt poses
                    if np.sum(pose) == 0:
                        continue

                    # add offset to x axis to seperate gt from pred.
                    # NOTE: use 1000 to filter out padded pose
                    threshold = 1000 + 1
                    xs = [pose[i][0] for i in range(14) if pose[i][2] > threshold]
                    ys = [-pose[i][1] for i in range(14) if pose[i][2] > threshold]
                    zs = [pose[i][2] for i in range(14) if pose[i][2] > threshold]

                    # put vertical axis y to third axis of matplotlib
                    ax.scatter(xs, zs, ys, s=10, c='blue', marker='o')

                    # draw skeleton lines
                    for (i, j) in pose_config.KPT_LINE_IDX:
                        if pose[i][2] <= threshold or pose[j][2] <= threshold:
                            continue
                        ax.plot([pose[i][0], pose[j][0]], [pose[i][2], pose[j][2]],
                                [-pose[i][1], -pose[j][1]],
                                color='blue', linestyle='-', linewidth=1)

            canvas.draw()  # draw the canvas, cache the renderer
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            cv2.imshow("pose_3d", image)
            plt.cla()  # clear image.

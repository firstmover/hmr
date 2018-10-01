"""
LYC: 
    get_beta.py is copy and modified from demo.py for generating shape descriptor, beta, from images. 

Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os 

import json 
import numpy as np
import skimage.io as io
import tensorflow as tf
from absl import flags

import src.config
from src.RunModel import RunModel
from src.util import image as img_util
from src.util import openpose as op_util
from src.util import renderer as vis_util

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, model, json_path=None):

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    # input_img: (224, 224, 3)
    # img: (224, 224, 3)

    # Add batch dimension: 1 x D x D x 3
    return model.predict(np.expand_dims(input_img, 0))


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    sess = tf.Session()
    model = RunModel(config, sess=sess)

    raw_img_dir = "/afs/csail.mit.edu/u/l/liuyingcheng/lyc_storage/rf-pose-shape/single_person"
    _, exp_dirs, _ = next(os.walk(raw_img_dir))
    for exp in exp_dirs:
        print("exp:", exp)
        exp_root = os.path.join(raw_img_dir, exp)
        crop_img_dir = os.path.join(exp_root, "crop_person")
        if not os.path.exists(crop_img_dir):
            raise ValueError("crop person image not found.")
        _, _, crop_img_names = next(os.walk(crop_img_dir))
        crop_img_names = [f for f in crop_img_names if (".png" in f or ".jpg" in f)]
        crop_img_pathes = [os.path.join(crop_img_dir, f) for f in crop_img_names]
        print("get nr crop img = {}".format(len(crop_img_pathes)))
        img_name2result = {name: main(img, model, None) for name, img in zip(crop_img_names, crop_img_pathes)}
        print("get nr result = {}".format(len(list(img_name2result.keys()))))
        with open(os.path.join(exp_root, "img_name2result.json"), "w") as f:
            f.write(json.dumps(img_name2result))


import torch.nn as nn
from ..utils import log
import comfy.model_management as mm
from comfy.utils import ProgressBar
from tqdm import tqdm

def update_transformer(transformer, state_dict):
    
    concat_dim = 4
    transformer.dwpose_embedding = nn.Sequential(
                nn.Conv3d(3, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,1,1), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, (3,3,3), stride=(1,2,2), padding=(1,1,1)),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2,2,2), padding=1),
                nn.SiLU(),
                nn.Conv3d(concat_dim * 4, 5120, (1,2,2), stride=(1,2,2), padding=0))

    randomref_dim = 20
    transformer.randomref_embedding_pose = nn.Sequential(
                nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
                )
    state_dict_new = {}
    for key in list(state_dict.keys()):
        if "dwpose_embedding" in key:
            state_dict_new[key.split("dwpose_embedding.")[1]] = state_dict.pop(key)
    transformer.dwpose_embedding.load_state_dict(state_dict_new, strict=True)
    state_dict_new = {}
    for key in list(state_dict.keys()):
        if "randomref_embedding_pose" in key:
            state_dict_new[key.split("randomref_embedding_pose.")[1]] = state_dict.pop(key)
    transformer.randomref_embedding_pose.load_state_dict(state_dict_new,strict=True)
    return transformer

# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
import torch
import numpy as np
import copy
import torch
import numpy as np
import math

from .dwpose.wholebody import Wholebody

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

class DWposeDetector:
    def __init__(self, model_det, model_pose):
        self.pose_estimation = Wholebody(model_det, model_pose)

    def __call__(self, oriImg, score_threshold=0.3):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > score_threshold:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<score_threshold
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > score_threshold:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) 
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) 
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)
            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # return draw_pose(pose, H, W)
            return pose

def draw_pose(pose, H, W, stick_width=4,draw_body=True, draw_hands=True, draw_feet=True, 
              body_keypoint_size=4, hand_keypoint_size=4, draw_head=True):
    from .dwpose.util import draw_body_and_foot, draw_handpose, draw_facepose
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_body_and_foot(canvas, candidate, subset, draw_body=draw_body, stick_width=stick_width, draw_feet=draw_feet, draw_head=draw_head, body_keypoint_size=body_keypoint_size)
    canvas = draw_handpose(canvas, hands, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size)
    canvas_without_face = copy.deepcopy(canvas)
    canvas = draw_facepose(canvas, faces)

    return canvas_without_face, canvas


def pose_extract(pose_images, ref_image, dwpose_model, height, width, score_threshold, stick_width,
                 draw_body=True, draw_hands=True, hand_keypoint_size=4, draw_feet=True,
                 body_keypoint_size=4, handle_not_detected="repeat", draw_head=True):
    
    results_vis = []
    comfy_pbar = ProgressBar(len(pose_images))

    if ref_image is not None:
        try:
            pose_ref = dwpose_model(ref_image.squeeze(0), score_threshold=score_threshold)
        except:
            raise ValueError("No pose detected in reference image")
    prev_pose = None
    for img in tqdm(pose_images, desc="Pose Extraction", unit="image", total=len(pose_images)):
        try:
            pose = dwpose_model(img, score_threshold=score_threshold)
            if handle_not_detected == "repeat":
                prev_pose = pose
        except:
            if prev_pose is not None:
                pose = prev_pose
            else:
                pose = np.zeros_like(img)
        results_vis.append(pose)
        comfy_pbar.update(1)
    
    bodies = results_vis[0]['bodies']
    faces = results_vis[0]['faces']
    hands = results_vis[0]['hands']
    candidate = bodies['candidate']

    if ref_image is not None:
        ref_bodies = pose_ref['bodies']
        ref_faces = pose_ref['faces']
        ref_hands = pose_ref['hands']
        ref_candidate = ref_bodies['candidate']


        ref_2_x = ref_candidate[2][0]
        ref_2_y = ref_candidate[2][1]
        ref_5_x = ref_candidate[5][0]
        ref_5_y = ref_candidate[5][1]
        ref_8_x = ref_candidate[8][0]
        ref_8_y = ref_candidate[8][1]
        ref_11_x = ref_candidate[11][0]
        ref_11_y = ref_candidate[11][1]
        ref_center1 = 0.5*(ref_candidate[2]+ref_candidate[5])
        ref_center2 = 0.5*(ref_candidate[8]+ref_candidate[11])

        zero_2_x = candidate[2][0]
        zero_2_y = candidate[2][1]
        zero_5_x = candidate[5][0]
        zero_5_y = candidate[5][1]
        zero_8_x = candidate[8][0]
        zero_8_y = candidate[8][1]
        zero_11_x = candidate[11][0]
        zero_11_y = candidate[11][1]
        zero_center1 = 0.5*(candidate[2]+candidate[5])
        zero_center2 = 0.5*(candidate[8]+candidate[11])

        x_ratio = (ref_5_x-ref_2_x)/(zero_5_x-zero_2_x)
        y_ratio = (ref_center2[1]-ref_center1[1])/(zero_center2[1]-zero_center1[1])

        results_vis[0]['bodies']['candidate'][:,0] *= x_ratio
        results_vis[0]['bodies']['candidate'][:,1] *= y_ratio
        results_vis[0]['faces'][:,:,0] *= x_ratio
        results_vis[0]['faces'][:,:,1] *= y_ratio
        results_vis[0]['hands'][:,:,0] *= x_ratio
        results_vis[0]['hands'][:,:,1] *= y_ratio
        
        ########neck########
        l_neck_ref = ((ref_candidate[0][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[0][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_neck_0 = ((candidate[0][0] - candidate[1][0]) ** 2 + (candidate[0][1] - candidate[1][1]) ** 2) ** 0.5
        neck_ratio = l_neck_ref / l_neck_0

        x_offset_neck = (candidate[1][0]-candidate[0][0])*(1.-neck_ratio)
        y_offset_neck = (candidate[1][1]-candidate[0][1])*(1.-neck_ratio)

        results_vis[0]['bodies']['candidate'][0,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][0,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][14,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][15,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_neck
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_neck
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_neck
        
        ########shoulder2########
        l_shoulder2_ref = ((ref_candidate[2][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[2][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder2_0 = ((candidate[2][0] - candidate[1][0]) ** 2 + (candidate[2][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder2_ratio = l_shoulder2_ref / l_shoulder2_0

        x_offset_shoulder2 = (candidate[1][0]-candidate[2][0])*(1.-shoulder2_ratio)
        y_offset_shoulder2 = (candidate[1][1]-candidate[2][1])*(1.-shoulder2_ratio)

        results_vis[0]['bodies']['candidate'][2,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][2,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_shoulder2
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_shoulder2
        results_vis[0]['hands'][1,:,0] += x_offset_shoulder2
        results_vis[0]['hands'][1,:,1] += y_offset_shoulder2

        ########shoulder5########
        l_shoulder5_ref = ((ref_candidate[5][0] - ref_candidate[1][0]) ** 2 + (ref_candidate[5][1] - ref_candidate[1][1]) ** 2) ** 0.5
        l_shoulder5_0 = ((candidate[5][0] - candidate[1][0]) ** 2 + (candidate[5][1] - candidate[1][1]) ** 2) ** 0.5

        shoulder5_ratio = l_shoulder5_ref / l_shoulder5_0

        x_offset_shoulder5 = (candidate[1][0]-candidate[5][0])*(1.-shoulder5_ratio)
        y_offset_shoulder5 = (candidate[1][1]-candidate[5][1])*(1.-shoulder5_ratio)

        results_vis[0]['bodies']['candidate'][5,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][5,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_shoulder5
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_shoulder5
        results_vis[0]['hands'][0,:,0] += x_offset_shoulder5
        results_vis[0]['hands'][0,:,1] += y_offset_shoulder5

        ########arm3########
        l_arm3_ref = ((ref_candidate[3][0] - ref_candidate[2][0]) ** 2 + (ref_candidate[3][1] - ref_candidate[2][1]) ** 2) ** 0.5
        l_arm3_0 = ((candidate[3][0] - candidate[2][0]) ** 2 + (candidate[3][1] - candidate[2][1]) ** 2) ** 0.5

        arm3_ratio = l_arm3_ref / l_arm3_0

        x_offset_arm3 = (candidate[2][0]-candidate[3][0])*(1.-arm3_ratio)
        y_offset_arm3 = (candidate[2][1]-candidate[3][1])*(1.-arm3_ratio)

        results_vis[0]['bodies']['candidate'][3,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][3,1] += y_offset_arm3
        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm3
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm3
        results_vis[0]['hands'][1,:,0] += x_offset_arm3
        results_vis[0]['hands'][1,:,1] += y_offset_arm3

        ########arm4########
        l_arm4_ref = ((ref_candidate[4][0] - ref_candidate[3][0]) ** 2 + (ref_candidate[4][1] - ref_candidate[3][1]) ** 2) ** 0.5
        l_arm4_0 = ((candidate[4][0] - candidate[3][0]) ** 2 + (candidate[4][1] - candidate[3][1]) ** 2) ** 0.5

        arm4_ratio = l_arm4_ref / l_arm4_0

        x_offset_arm4 = (candidate[3][0]-candidate[4][0])*(1.-arm4_ratio)
        y_offset_arm4 = (candidate[3][1]-candidate[4][1])*(1.-arm4_ratio)

        results_vis[0]['bodies']['candidate'][4,0] += x_offset_arm4
        results_vis[0]['bodies']['candidate'][4,1] += y_offset_arm4
        results_vis[0]['hands'][1,:,0] += x_offset_arm4
        results_vis[0]['hands'][1,:,1] += y_offset_arm4

        ########arm6########
        l_arm6_ref = ((ref_candidate[6][0] - ref_candidate[5][0]) ** 2 + (ref_candidate[6][1] - ref_candidate[5][1]) ** 2) ** 0.5
        l_arm6_0 = ((candidate[6][0] - candidate[5][0]) ** 2 + (candidate[6][1] - candidate[5][1]) ** 2) ** 0.5

        arm6_ratio = l_arm6_ref / l_arm6_0

        x_offset_arm6 = (candidate[5][0]-candidate[6][0])*(1.-arm6_ratio)
        y_offset_arm6 = (candidate[5][1]-candidate[6][1])*(1.-arm6_ratio)

        results_vis[0]['bodies']['candidate'][6,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][6,1] += y_offset_arm6
        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm6
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm6
        results_vis[0]['hands'][0,:,0] += x_offset_arm6
        results_vis[0]['hands'][0,:,1] += y_offset_arm6

        ########arm7########
        l_arm7_ref = ((ref_candidate[7][0] - ref_candidate[6][0]) ** 2 + (ref_candidate[7][1] - ref_candidate[6][1]) ** 2) ** 0.5
        l_arm7_0 = ((candidate[7][0] - candidate[6][0]) ** 2 + (candidate[7][1] - candidate[6][1]) ** 2) ** 0.5

        arm7_ratio = l_arm7_ref / l_arm7_0

        x_offset_arm7 = (candidate[6][0]-candidate[7][0])*(1.-arm7_ratio)
        y_offset_arm7 = (candidate[6][1]-candidate[7][1])*(1.-arm7_ratio)

        results_vis[0]['bodies']['candidate'][7,0] += x_offset_arm7
        results_vis[0]['bodies']['candidate'][7,1] += y_offset_arm7
        results_vis[0]['hands'][0,:,0] += x_offset_arm7
        results_vis[0]['hands'][0,:,1] += y_offset_arm7

        ########head14########
        l_head14_ref = ((ref_candidate[14][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[14][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head14_0 = ((candidate[14][0] - candidate[0][0]) ** 2 + (candidate[14][1] - candidate[0][1]) ** 2) ** 0.5

        head14_ratio = l_head14_ref / l_head14_0

        x_offset_head14 = (candidate[0][0]-candidate[14][0])*(1.-head14_ratio)
        y_offset_head14 = (candidate[0][1]-candidate[14][1])*(1.-head14_ratio)

        results_vis[0]['bodies']['candidate'][14,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][14,1] += y_offset_head14
        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head14
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head14

        ########head15########
        l_head15_ref = ((ref_candidate[15][0] - ref_candidate[0][0]) ** 2 + (ref_candidate[15][1] - ref_candidate[0][1]) ** 2) ** 0.5
        l_head15_0 = ((candidate[15][0] - candidate[0][0]) ** 2 + (candidate[15][1] - candidate[0][1]) ** 2) ** 0.5

        head15_ratio = l_head15_ref / l_head15_0

        x_offset_head15 = (candidate[0][0]-candidate[15][0])*(1.-head15_ratio)
        y_offset_head15 = (candidate[0][1]-candidate[15][1])*(1.-head15_ratio)

        results_vis[0]['bodies']['candidate'][15,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][15,1] += y_offset_head15
        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head15
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head15

        ########head16########
        l_head16_ref = ((ref_candidate[16][0] - ref_candidate[14][0]) ** 2 + (ref_candidate[16][1] - ref_candidate[14][1]) ** 2) ** 0.5
        l_head16_0 = ((candidate[16][0] - candidate[14][0]) ** 2 + (candidate[16][1] - candidate[14][1]) ** 2) ** 0.5

        head16_ratio = l_head16_ref / l_head16_0

        x_offset_head16 = (candidate[14][0]-candidate[16][0])*(1.-head16_ratio)
        y_offset_head16 = (candidate[14][1]-candidate[16][1])*(1.-head16_ratio)

        results_vis[0]['bodies']['candidate'][16,0] += x_offset_head16
        results_vis[0]['bodies']['candidate'][16,1] += y_offset_head16

        ########head17########
        l_head17_ref = ((ref_candidate[17][0] - ref_candidate[15][0]) ** 2 + (ref_candidate[17][1] - ref_candidate[15][1]) ** 2) ** 0.5
        l_head17_0 = ((candidate[17][0] - candidate[15][0]) ** 2 + (candidate[17][1] - candidate[15][1]) ** 2) ** 0.5

        head17_ratio = l_head17_ref / l_head17_0

        x_offset_head17 = (candidate[15][0]-candidate[17][0])*(1.-head17_ratio)
        y_offset_head17 = (candidate[15][1]-candidate[17][1])*(1.-head17_ratio)

        results_vis[0]['bodies']['candidate'][17,0] += x_offset_head17
        results_vis[0]['bodies']['candidate'][17,1] += y_offset_head17
        
        ########MovingAverage########
        
        ########left leg########
        l_ll1_ref = ((ref_candidate[8][0] - ref_candidate[9][0]) ** 2 + (ref_candidate[8][1] - ref_candidate[9][1]) ** 2) ** 0.5
        l_ll1_0 = ((candidate[8][0] - candidate[9][0]) ** 2 + (candidate[8][1] - candidate[9][1]) ** 2) ** 0.5
        ll1_ratio = l_ll1_ref / l_ll1_0

        x_offset_ll1 = (candidate[9][0]-candidate[8][0])*(ll1_ratio-1.)
        y_offset_ll1 = (candidate[9][1]-candidate[8][1])*(ll1_ratio-1.)

        results_vis[0]['bodies']['candidate'][9,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][9,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll1
        results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll1
        results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll1

        l_ll2_ref = ((ref_candidate[9][0] - ref_candidate[10][0]) ** 2 + (ref_candidate[9][1] - ref_candidate[10][1]) ** 2) ** 0.5
        l_ll2_0 = ((candidate[9][0] - candidate[10][0]) ** 2 + (candidate[9][1] - candidate[10][1]) ** 2) ** 0.5
        ll2_ratio = l_ll2_ref / l_ll2_0

        x_offset_ll2 = (candidate[10][0]-candidate[9][0])*(ll2_ratio-1.)
        y_offset_ll2 = (candidate[10][1]-candidate[9][1])*(ll2_ratio-1.)

        results_vis[0]['bodies']['candidate'][10,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][10,1] += y_offset_ll2
        results_vis[0]['bodies']['candidate'][19,0] += x_offset_ll2
        results_vis[0]['bodies']['candidate'][19,1] += y_offset_ll2

        ########right leg########
        l_rl1_ref = ((ref_candidate[11][0] - ref_candidate[12][0]) ** 2 + (ref_candidate[11][1] - ref_candidate[12][1]) ** 2) ** 0.5
        l_rl1_0 = ((candidate[11][0] - candidate[12][0]) ** 2 + (candidate[11][1] - candidate[12][1]) ** 2) ** 0.5
        rl1_ratio = l_rl1_ref / l_rl1_0

        x_offset_rl1 = (candidate[12][0]-candidate[11][0])*(rl1_ratio-1.)
        y_offset_rl1 = (candidate[12][1]-candidate[11][1])*(rl1_ratio-1.)

        results_vis[0]['bodies']['candidate'][12,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][12,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl1
        results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl1
        results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl1

        l_rl2_ref = ((ref_candidate[12][0] - ref_candidate[13][0]) ** 2 + (ref_candidate[12][1] - ref_candidate[13][1]) ** 2) ** 0.5
        l_rl2_0 = ((candidate[12][0] - candidate[13][0]) ** 2 + (candidate[12][1] - candidate[13][1]) ** 2) ** 0.5
        rl2_ratio = l_rl2_ref / l_rl2_0

        x_offset_rl2 = (candidate[13][0]-candidate[12][0])*(rl2_ratio-1.)
        y_offset_rl2 = (candidate[13][1]-candidate[12][1])*(rl2_ratio-1.)

        results_vis[0]['bodies']['candidate'][13,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][13,1] += y_offset_rl2
        results_vis[0]['bodies']['candidate'][18,0] += x_offset_rl2
        results_vis[0]['bodies']['candidate'][18,1] += y_offset_rl2

        offset = ref_candidate[1] - results_vis[0]['bodies']['candidate'][1]

        results_vis[0]['bodies']['candidate'] += offset[np.newaxis, :]
        results_vis[0]['faces'] += offset[np.newaxis, np.newaxis, :]
        results_vis[0]['hands'] += offset[np.newaxis, np.newaxis, :]

        for i in range(1, len(results_vis)):
            results_vis[i]['bodies']['candidate'][:,0] *= x_ratio
            results_vis[i]['bodies']['candidate'][:,1] *= y_ratio
            results_vis[i]['faces'][:,:,0] *= x_ratio
            results_vis[i]['faces'][:,:,1] *= y_ratio
            results_vis[i]['hands'][:,:,0] *= x_ratio
            results_vis[i]['hands'][:,:,1] *= y_ratio

            ########neck########
            x_offset_neck = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][0][0])*(1.-neck_ratio)
            y_offset_neck = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][0][1])*(1.-neck_ratio)

            results_vis[i]['bodies']['candidate'][0,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][0,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][14,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][15,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_neck
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_neck
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_neck

            ########shoulder2########
            

            x_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][2][0])*(1.-shoulder2_ratio)
            y_offset_shoulder2 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][2][1])*(1.-shoulder2_ratio)

            results_vis[i]['bodies']['candidate'][2,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][2,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_shoulder2
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_shoulder2
            results_vis[i]['hands'][1,:,0] += x_offset_shoulder2
            results_vis[i]['hands'][1,:,1] += y_offset_shoulder2

            ########shoulder5########

            x_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][0]-results_vis[i]['bodies']['candidate'][5][0])*(1.-shoulder5_ratio)
            y_offset_shoulder5 = (results_vis[i]['bodies']['candidate'][1][1]-results_vis[i]['bodies']['candidate'][5][1])*(1.-shoulder5_ratio)

            results_vis[i]['bodies']['candidate'][5,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][5,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_shoulder5
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_shoulder5
            results_vis[i]['hands'][0,:,0] += x_offset_shoulder5
            results_vis[i]['hands'][0,:,1] += y_offset_shoulder5

            ########arm3########

            x_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][0]-results_vis[i]['bodies']['candidate'][3][0])*(1.-arm3_ratio)
            y_offset_arm3 = (results_vis[i]['bodies']['candidate'][2][1]-results_vis[i]['bodies']['candidate'][3][1])*(1.-arm3_ratio)

            results_vis[i]['bodies']['candidate'][3,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][3,1] += y_offset_arm3
            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm3
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm3
            results_vis[i]['hands'][1,:,0] += x_offset_arm3
            results_vis[i]['hands'][1,:,1] += y_offset_arm3

            ########arm4########

            x_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][0]-results_vis[i]['bodies']['candidate'][4][0])*(1.-arm4_ratio)
            y_offset_arm4 = (results_vis[i]['bodies']['candidate'][3][1]-results_vis[i]['bodies']['candidate'][4][1])*(1.-arm4_ratio)

            results_vis[i]['bodies']['candidate'][4,0] += x_offset_arm4
            results_vis[i]['bodies']['candidate'][4,1] += y_offset_arm4
            results_vis[i]['hands'][1,:,0] += x_offset_arm4
            results_vis[i]['hands'][1,:,1] += y_offset_arm4

            ########arm6########

            x_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][0]-results_vis[i]['bodies']['candidate'][6][0])*(1.-arm6_ratio)
            y_offset_arm6 = (results_vis[i]['bodies']['candidate'][5][1]-results_vis[i]['bodies']['candidate'][6][1])*(1.-arm6_ratio)

            results_vis[i]['bodies']['candidate'][6,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][6,1] += y_offset_arm6
            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm6
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm6
            results_vis[i]['hands'][0,:,0] += x_offset_arm6
            results_vis[i]['hands'][0,:,1] += y_offset_arm6

            ########arm7########

            x_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][0]-results_vis[i]['bodies']['candidate'][7][0])*(1.-arm7_ratio)
            y_offset_arm7 = (results_vis[i]['bodies']['candidate'][6][1]-results_vis[i]['bodies']['candidate'][7][1])*(1.-arm7_ratio)

            results_vis[i]['bodies']['candidate'][7,0] += x_offset_arm7
            results_vis[i]['bodies']['candidate'][7,1] += y_offset_arm7
            results_vis[i]['hands'][0,:,0] += x_offset_arm7
            results_vis[i]['hands'][0,:,1] += y_offset_arm7

            ########head14########

            x_offset_head14 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][14][0])*(1.-head14_ratio)
            y_offset_head14 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][14][1])*(1.-head14_ratio)

            results_vis[i]['bodies']['candidate'][14,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][14,1] += y_offset_head14
            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head14
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head14

            ########head15########

            x_offset_head15 = (results_vis[i]['bodies']['candidate'][0][0]-results_vis[i]['bodies']['candidate'][15][0])*(1.-head15_ratio)
            y_offset_head15 = (results_vis[i]['bodies']['candidate'][0][1]-results_vis[i]['bodies']['candidate'][15][1])*(1.-head15_ratio)

            results_vis[i]['bodies']['candidate'][15,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][15,1] += y_offset_head15
            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head15
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head15

            ########head16########

            x_offset_head16 = (results_vis[i]['bodies']['candidate'][14][0]-results_vis[i]['bodies']['candidate'][16][0])*(1.-head16_ratio)
            y_offset_head16 = (results_vis[i]['bodies']['candidate'][14][1]-results_vis[i]['bodies']['candidate'][16][1])*(1.-head16_ratio)

            results_vis[i]['bodies']['candidate'][16,0] += x_offset_head16
            results_vis[i]['bodies']['candidate'][16,1] += y_offset_head16

            ########head17########
            x_offset_head17 = (results_vis[i]['bodies']['candidate'][15][0]-results_vis[i]['bodies']['candidate'][17][0])*(1.-head17_ratio)
            y_offset_head17 = (results_vis[i]['bodies']['candidate'][15][1]-results_vis[i]['bodies']['candidate'][17][1])*(1.-head17_ratio)

            results_vis[i]['bodies']['candidate'][17,0] += x_offset_head17
            results_vis[i]['bodies']['candidate'][17,1] += y_offset_head17

            # ########MovingAverage########

            ########left leg########
            x_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][0]-results_vis[i]['bodies']['candidate'][8][0])*(ll1_ratio-1.)
            y_offset_ll1 = (results_vis[i]['bodies']['candidate'][9][1]-results_vis[i]['bodies']['candidate'][8][1])*(ll1_ratio-1.)

            results_vis[i]['bodies']['candidate'][9,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][9,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll1
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll1
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll1



            x_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][0]-results_vis[i]['bodies']['candidate'][9][0])*(ll2_ratio-1.)
            y_offset_ll2 = (results_vis[i]['bodies']['candidate'][10][1]-results_vis[i]['bodies']['candidate'][9][1])*(ll2_ratio-1.)

            results_vis[i]['bodies']['candidate'][10,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][10,1] += y_offset_ll2
            results_vis[i]['bodies']['candidate'][19,0] += x_offset_ll2
            results_vis[i]['bodies']['candidate'][19,1] += y_offset_ll2

            ########right leg########

            x_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][0]-results_vis[i]['bodies']['candidate'][11][0])*(rl1_ratio-1.)
            y_offset_rl1 = (results_vis[i]['bodies']['candidate'][12][1]-results_vis[i]['bodies']['candidate'][11][1])*(rl1_ratio-1.)

            results_vis[i]['bodies']['candidate'][12,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][12,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl1
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl1
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl1


            x_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][0]-results_vis[i]['bodies']['candidate'][12][0])*(rl2_ratio-1.)
            y_offset_rl2 = (results_vis[i]['bodies']['candidate'][13][1]-results_vis[i]['bodies']['candidate'][12][1])*(rl2_ratio-1.)

            results_vis[i]['bodies']['candidate'][13,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][13,1] += y_offset_rl2
            results_vis[i]['bodies']['candidate'][18,0] += x_offset_rl2
            results_vis[i]['bodies']['candidate'][18,1] += y_offset_rl2

            results_vis[i]['bodies']['candidate'] += offset[np.newaxis, :]
            results_vis[i]['faces'] += offset[np.newaxis, np.newaxis, :]
            results_vis[i]['hands'] += offset[np.newaxis, np.newaxis, :]
    
    dwpose_woface_list = []
    for i in range(len(results_vis)):
        #try:
        dwpose_woface, dwpose_wface = draw_pose(results_vis[i], H=height, W=width, stick_width=stick_width,
                                                    draw_body=draw_body, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size,
                                                    draw_feet=draw_feet, body_keypoint_size=body_keypoint_size, draw_head=draw_head)
        result = torch.from_numpy(dwpose_woface)
        #except:
        #    result = torch.zeros((height, width, 3), dtype=torch.uint8)
        dwpose_woface_list.append(result)
    dwpose_woface_tensor = torch.stack(dwpose_woface_list, dim=0)

    dwpose_woface_ref_tensor = None
    if ref_image is not None:
        dwpose_woface_ref, dwpose_wface_ref = draw_pose(pose_ref, H=height, W=width, stick_width=stick_width,
                                                        draw_body=draw_body, draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size,
                                                        draw_feet=draw_feet, body_keypoint_size=body_keypoint_size, draw_head=draw_head)
        dwpose_woface_ref_tensor = torch.from_numpy(dwpose_woface_ref)

    return dwpose_woface_tensor, dwpose_woface_ref_tensor


def pose_keypoint_to_dwpose_format(pose_keypoint, canvas_width, canvas_height):
    """
    Convert POSE_KEYPOINT format to DWPose internal format
    
    Args:
        pose_keypoint: POSE_KEYPOINT data (list of dicts)
        canvas_width: Canvas width for coordinate conversion
        canvas_height: Canvas height for coordinate conversion
    
    Returns:
        dict: DWPose format with 'bodies', 'faces', 'hands' keys
    """
    import numpy as np
    
    if not pose_keypoint or len(pose_keypoint) == 0:
        return {'bodies': {'candidate': np.array([]), 'subset': np.array([])}, 'faces': np.array([]), 'hands': np.array([])}
    
    frame_data = pose_keypoint[0]  # Use first frame
    people = frame_data.get('people', [])
    
    if len(people) == 0:
        return {'bodies': {'candidate': np.array([]), 'subset': np.array([])}, 'faces': np.array([]), 'hands': np.array([])}
    
    person = people[0]  # Use first person
    
    # Extract keypoints
    body_kpts = person.get('pose_keypoints_2d', [])
    face_kpts = person.get('face_keypoints_2d', [])
    lhand_kpts = person.get('hand_left_keypoints_2d', [])
    rhand_kpts = person.get('hand_right_keypoints_2d', [])
    
    # Convert body keypoints to candidate format
    candidates = []
    subset = []
    
    if body_kpts and len(body_kpts) >= 75:  # 25 points * 3 (x,y,conf)
        for i in range(25):  # Support full 25 points including toes
            x = body_kpts[i*3] / canvas_width if canvas_width > 0 else body_kpts[i*3]
            y = body_kpts[i*3+1] / canvas_height if canvas_height > 0 else body_kpts[i*3+1]
            conf = body_kpts[i*3+2]
            candidates.append([x, y, conf])
        
        # Create subset (which keypoints are valid)
        subset_row = []
        for i in range(25):
            if body_kpts[i*3+2] > 0:  # confidence > 0
                subset_row.append(i)
            else:
                subset_row.append(-1)
        subset.append(subset_row)
    
    candidate_array = np.array(candidates) if candidates else np.array([])
    subset_array = np.array(subset) if subset else np.array([])
    
    # Convert face keypoints
    faces = []
    if face_kpts and len(face_kpts) >= 210:  # 70 points * 3
        face_points = []
        for i in range(70):
            x = face_kpts[i*3] / canvas_width if canvas_width > 0 else face_kpts[i*3]
            y = face_kpts[i*3+1] / canvas_height if canvas_height > 0 else face_kpts[i*3+1]
            conf = face_kpts[i*3+2]
            face_points.append([x, y, conf])
        faces.append(face_points)
    
    # Convert hand keypoints
    hands = []
    if lhand_kpts and len(lhand_kpts) >= 63:  # 21 points * 3
        lhand_points = []
        for i in range(21):
            x = lhand_kpts[i*3] / canvas_width if canvas_width > 0 else lhand_kpts[i*3]
            y = lhand_kpts[i*3+1] / canvas_height if canvas_height > 0 else lhand_kpts[i*3+1]
            conf = lhand_kpts[i*3+2]
            lhand_points.append([x, y, conf])
        hands.append(lhand_points)
    
    if rhand_kpts and len(rhand_kpts) >= 63:  # 21 points * 3
        rhand_points = []
        for i in range(21):
            x = rhand_kpts[i*3] / canvas_width if canvas_width > 0 else rhand_kpts[i*3]
            y = rhand_kpts[i*3+1] / canvas_height if canvas_height > 0 else rhand_kpts[i*3+1]
            conf = rhand_kpts[i*3+2]
            rhand_points.append([x, y, conf])
        hands.append(rhand_points)
    
    faces_array = np.array(faces) if faces else np.array([])
    hands_array = np.array(hands) if hands else np.array([])
    
    return {
        'bodies': {
            'candidate': candidate_array,
            'subset': subset_array
        },
        'faces': faces_array,
        'hands': hands_array
    }


def dwpose_format_to_pose_keypoint(candidate, faces, hands, canvas_width, canvas_height):
    """
    Convert DWPose internal format back to POSE_KEYPOINT format
    
    Args:
        candidate: Body keypoints in DWPose format
        faces: Face keypoints in DWPose format  
        hands: Hand keypoints in DWPose format
        canvas_width: Canvas width for coordinate conversion
        canvas_height: Canvas height for coordinate conversion
    
    Returns:
        list: POSE_KEYPOINT format data
    """
    import numpy as np
    
    # Convert body keypoints
    body_keypoints = []
    if len(candidate) > 0:
        for i in range(min(25, len(candidate))):  # Support up to 25 points including toes
            x = candidate[i][0] * canvas_width if canvas_width > 0 else candidate[i][0]
            y = candidate[i][1] * canvas_height if canvas_height > 0 else candidate[i][1]
            conf = candidate[i][2] if len(candidate[i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
            body_keypoints.extend([x, y, conf])
    
    # Pad to 25 points if needed
    while len(body_keypoints) < 75:  # 25 * 3
        body_keypoints.extend([0.0, 0.0, 0.0])
    
    # Convert face keypoints
    face_keypoints = []
    if len(faces) > 0 and len(faces[0]) > 0:
        for i in range(min(70, len(faces[0]))):
            x = faces[0][i][0] * canvas_width if canvas_width > 0 else faces[0][i][0]
            y = faces[0][i][1] * canvas_height if canvas_height > 0 else faces[0][i][1]
            conf = faces[0][i][2] if len(faces[0][i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
            face_keypoints.extend([x, y, conf])
    
    # Pad to 70 points if needed
    while len(face_keypoints) < 210:  # 70 * 3
        face_keypoints.extend([0.0, 0.0, 0.0])
    
    # Convert hand keypoints
    lhand_keypoints = []
    rhand_keypoints = []
    
    if len(hands) > 0:
        # Left hand
        if len(hands[0]) > 0:
            for i in range(min(21, len(hands[0]))):
                x = hands[0][i][0] * canvas_width if canvas_width > 0 else hands[0][i][0]
                y = hands[0][i][1] * canvas_height if canvas_height > 0 else hands[0][i][1]
                conf = hands[0][i][2] if len(hands[0][i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
                lhand_keypoints.extend([x, y, conf])
        
        # Right hand
        if len(hands) > 1 and len(hands[1]) > 0:
            for i in range(min(21, len(hands[1]))):
                x = hands[1][i][0] * canvas_width if canvas_width > 0 else hands[1][i][0]
                y = hands[1][i][1] * canvas_height if canvas_height > 0 else hands[1][i][1]
                conf = hands[1][i][2] if len(hands[1][i]) > 2 else 1.0  # Use 1.0 instead of 0.0 for missing confidence
                rhand_keypoints.extend([x, y, conf])
    
    # Pad hand keypoints to 21 points if needed
    while len(lhand_keypoints) < 63:  # 21 * 3
        lhand_keypoints.extend([0.0, 0.0, 0.0])
    while len(rhand_keypoints) < 63:  # 21 * 3
        rhand_keypoints.extend([0.0, 0.0, 0.0])
    
    # Create POSE_KEYPOINT structure
    person_data = {
        "pose_keypoints_2d": body_keypoints,
        "face_keypoints_2d": face_keypoints,
        "hand_left_keypoints_2d": lhand_keypoints,
        "hand_right_keypoints_2d": rhand_keypoints
    }
    
    frame_data = {
        "version": "1.0",
        "people": [person_data] if len(candidate) > 0 else [],
        "canvas_width": canvas_width,
        "canvas_height": canvas_height
    }
    
    return frame_data


def get_input_from_references(pose_keypoint, reference_pose, face_eye_correction=True):
        """
        Calculate how to resize the given pose_keypoint based on reference pose
        
        Args:
            pose_keypoint: Current POSE_KEYPOINT to resize
            reference_pose: Reference POSE_KEYPOINT with target proportions
            face_eye_correction: Whether to apply face eye correction
        
        Returns:
            Dict with transformation parameters for resizing
        """
        import math
        
        if not reference_pose or len(reference_pose) == 0:
            return None
        
        ref_frame = reference_pose[0]
        ref_people = ref_frame.get('people', [])
        
        if len(ref_people) == 0:
            return None
            
        ref_person = ref_people[0]
        ref_body_kpts = ref_person.get('pose_keypoints_2d', [])
        ref_face_kpts = ref_person.get('face_keypoints_2d', [])
        
        # Extract reference keypoints (assuming normalized coordinates)
        if len(ref_body_kpts) < 75:  # 25 * 3
            return None
            
        ref_points = []
        for i in range(25):
            x = ref_body_kpts[i*3]
            y = ref_body_kpts[i*3+1]
            conf = ref_body_kpts[i*3+2]
            ref_points.append([x, y, conf])
        
        # Eye distance calculation (DWPose keypoints 14, 15)
        ref_right_eye = ref_points[14]  # Right eye
        ref_left_eye = ref_points[15]   # Left eye
        
        if ref_right_eye[2] <= 0 or ref_left_eye[2] <= 0:
            return None
        
        ref_eye_distance = math.sqrt((ref_right_eye[0] - ref_left_eye[0])**2 + 
                                   (ref_right_eye[1] - ref_left_eye[1])**2)
        
        if ref_eye_distance <= 0:
            return None
        
        # If face_eye_correction is enabled, try to get more accurate eye positions from face keypoints
        if face_eye_correction and len(ref_face_kpts) >= 210:  # 70 * 3
            # Face keypoints: right eye center around index 36-41, left eye center around 42-47
            # We'll use approximate centers
            ref_face_points = []
            for i in range(70):
                x = ref_face_kpts[i*3]
                y = ref_face_kpts[i*3+1]
                conf = ref_face_kpts[i*3+2]
                ref_face_points.append([x, y, conf])
            
            # Calculate more precise eye centers from face keypoints if available
            try:
                # Right eye center (face keypoints ~36-41)
                right_eye_x = sum(ref_face_points[i][0] for i in range(36, 42) if ref_face_points[i][2] > 0) / 6
                right_eye_y = sum(ref_face_points[i][1] for i in range(36, 42) if ref_face_points[i][2] > 0) / 6
                
                # Left eye center (face keypoints ~42-47) 
                left_eye_x = sum(ref_face_points[i][0] for i in range(42, 48) if ref_face_points[i][2] > 0) / 6
                left_eye_y = sum(ref_face_points[i][1] for i in range(42, 48) if ref_face_points[i][2] > 0) / 6
                
                face_eye_distance = math.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
                
                if face_eye_distance > 0:
                    ref_eye_distance = face_eye_distance
                    ref_right_eye = [right_eye_x, right_eye_y, 1.0]
                    ref_left_eye = [left_eye_x, left_eye_y, 1.0]
                    
            except (IndexError, ZeroDivisionError):
                # Fall back to body keypoints if face processing fails
                pass
        
        return {
            'eye_distance': ref_eye_distance,
            'right_eye': ref_right_eye,
            'left_eye': ref_left_eye,
            'reference_points': ref_points
        }


def process_proportions_with_reference(pose_keypoint, reference_data, score_threshold=0.3, face_eye_correction=True):
    """
    Process pose proportions using reference data
    
    Args:
        pose_keypoint: Current pose keypoint data to modify
        reference_data: Reference transformation data from get_input_from_references()
        score_threshold: Confidence threshold for processing keypoints  
        face_eye_correction: Whether face eye correction was used in reference
    
    Returns:
        Modified pose_keypoint with adjusted proportions
    """
    import math
    import copy
    
    if not pose_keypoint or len(pose_keypoint) == 0 or not reference_data:
        return pose_keypoint
    
    frame_data = pose_keypoint[0]
    people = frame_data.get('people', [])
    
    if len(people) == 0:
        return pose_keypoint
    
    person = people[0]
    body_kpts = person.get('pose_keypoints_2d', [])
    face_kpts = person.get('face_keypoints_2d', [])
    
    if len(body_kpts) < 75:  # 25 * 3
        return pose_keypoint
    
    # Extract current pose points
    current_points = []
    for i in range(25):
        x = body_kpts[i*3]
        y = body_kpts[i*3+1]
        conf = body_kpts[i*3+2]
        current_points.append([x, y, conf])
    
    # Get current eye positions (DWPose keypoints 14, 15)
    current_right_eye = current_points[14]  # Right eye
    current_left_eye = current_points[15]   # Left eye
    
    if current_right_eye[2] <= score_threshold or current_left_eye[2] <= score_threshold:
        return pose_keypoint
    
    current_eye_distance = math.sqrt((current_right_eye[0] - current_left_eye[0])**2 + 
                                   (current_right_eye[1] - current_left_eye[1])**2)
    
    if current_eye_distance <= 0:
        return pose_keypoint
    
    # Apply face eye correction if enabled
    if face_eye_correction and len(face_kpts) >= 210:  # 70 * 3
        current_face_points = []
        for i in range(70):
            x = face_kpts[i*3]
            y = face_kpts[i*3+1]
            conf = face_kpts[i*3+2]
            current_face_points.append([x, y, conf])
        
        try:
            # Calculate face-based eye centers
            right_eye_x = sum(current_face_points[i][0] for i in range(36, 42) if current_face_points[i][2] > score_threshold) / 6
            right_eye_y = sum(current_face_points[i][1] for i in range(36, 42) if current_face_points[i][2] > score_threshold) / 6
            
            left_eye_x = sum(current_face_points[i][0] for i in range(42, 48) if current_face_points[i][2] > score_threshold) / 6
            left_eye_y = sum(current_face_points[i][1] for i in range(42, 48) if current_face_points[i][2] > score_threshold) / 6
            
            face_eye_distance = math.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
            
            if face_eye_distance > 0:
                current_eye_distance = face_eye_distance
                current_right_eye = [right_eye_x, right_eye_y, 1.0]
                current_left_eye = [left_eye_x, left_eye_y, 1.0]
                
        except (IndexError, ZeroDivisionError):
            # Fall back to body keypoints if face processing fails
            pass
    
    # Calculate scaling factor based on eye distance
    scale_factor = reference_data['eye_distance'] / current_eye_distance
    
    # Calculate eye center for scaling origin
    current_eye_center_x = (current_right_eye[0] + current_left_eye[0]) / 2
    current_eye_center_y = (current_right_eye[1] + current_left_eye[1]) / 2
    
    # Apply scaling to all body keypoints
    modified_body_kpts = copy.deepcopy(body_kpts)
    
    for i in range(25):
        idx = i * 3
        if body_kpts[idx + 2] > score_threshold:  # confidence check
            # Scale relative to eye center
            scaled_x = current_eye_center_x + (body_kpts[idx] - current_eye_center_x) * scale_factor
            scaled_y = current_eye_center_y + (body_kpts[idx + 1] - current_eye_center_y) * scale_factor
            
            modified_body_kpts[idx] = scaled_x
            modified_body_kpts[idx + 1] = scaled_y
    
    # Apply same scaling to hands and face keypoints
    modified_face_kpts = copy.deepcopy(face_kpts)
    for i in range(70):
        idx = i * 3
        if len(face_kpts) > idx + 2 and face_kpts[idx + 2] > score_threshold:
            scaled_x = current_eye_center_x + (face_kpts[idx] - current_eye_center_x) * scale_factor
            scaled_y = current_eye_center_y + (face_kpts[idx + 1] - current_eye_center_y) * scale_factor
            
            modified_face_kpts[idx] = scaled_x
            modified_face_kpts[idx + 1] = scaled_y
    
    # Apply scaling to hand keypoints
    lhand_kpts = person.get('hand_left_keypoints_2d', [])
    rhand_kpts = person.get('hand_right_keypoints_2d', [])
    
    modified_lhand_kpts = copy.deepcopy(lhand_kpts)
    for i in range(21):
        idx = i * 3
        if len(lhand_kpts) > idx + 2 and lhand_kpts[idx + 2] > score_threshold:
            scaled_x = current_eye_center_x + (lhand_kpts[idx] - current_eye_center_x) * scale_factor
            scaled_y = current_eye_center_y + (lhand_kpts[idx + 1] - current_eye_center_y) * scale_factor
            
            modified_lhand_kpts[idx] = scaled_x
            modified_lhand_kpts[idx + 1] = scaled_y
    
    modified_rhand_kpts = copy.deepcopy(rhand_kpts)
    for i in range(21):
        idx = i * 3
        if len(rhand_kpts) > idx + 2 and rhand_kpts[idx + 2] > score_threshold:
            scaled_x = current_eye_center_x + (rhand_kpts[idx] - current_eye_center_x) * scale_factor
            scaled_y = current_eye_center_y + (rhand_kpts[idx + 1] - current_eye_center_y) * scale_factor
            
            modified_rhand_kpts[idx] = scaled_x
            modified_rhand_kpts[idx + 1] = scaled_y
    
    # Create modified pose keypoint data
    modified_pose = copy.deepcopy(pose_keypoint)
    modified_pose[0]['people'][0]['pose_keypoints_2d'] = modified_body_kpts
    modified_pose[0]['people'][0]['face_keypoints_2d'] = modified_face_kpts
    modified_pose[0]['people'][0]['hand_left_keypoints_2d'] = modified_lhand_kpts
    modified_pose[0]['people'][0]['hand_right_keypoints_2d'] = modified_rhand_kpts
    
    return modified_pose


def normalize_pose_keypoint_coordinates(pose_keypoint, target_canvas_width=1.0, target_canvas_height=1.0):
    """
    Normalize POSE_KEYPOINT coordinates to specified canvas dimensions
    
    Args:
        pose_keypoint: POSE_KEYPOINT data to normalize
        target_canvas_width: Target canvas width (1.0 for normalized coordinates)
        target_canvas_height: Target canvas height (1.0 for normalized coordinates)
    
    Returns:
        Normalized pose_keypoint data
    """
    import copy
    
    if not pose_keypoint or len(pose_keypoint) == 0:
        return pose_keypoint
    
    frame_data = pose_keypoint[0]
    current_width = frame_data.get('canvas_width', 1.0)
    current_height = frame_data.get('canvas_height', 1.0)
    
    # No need to normalize if already at target dimensions
    if current_width == target_canvas_width and current_height == target_canvas_height:
        return pose_keypoint
    
    # Calculate scale factors
    scale_x = target_canvas_width / current_width if current_width > 0 else 1.0
    scale_y = target_canvas_height / current_height if current_height > 0 else 1.0
    
    normalized_pose = copy.deepcopy(pose_keypoint)
    
    for person in normalized_pose[0].get('people', []):
        # Normalize body keypoints
        body_kpts = person.get('pose_keypoints_2d', [])
        for i in range(0, len(body_kpts), 3):
            if i + 1 < len(body_kpts):
                body_kpts[i] *= scale_x        # x coordinate
                body_kpts[i+1] *= scale_y      # y coordinate
                # confidence stays the same
        
        # Normalize face keypoints
        face_kpts = person.get('face_keypoints_2d', [])
        for i in range(0, len(face_kpts), 3):
            if i + 1 < len(face_kpts):
                face_kpts[i] *= scale_x
                face_kpts[i+1] *= scale_y
        
        # Normalize hand keypoints
        for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            hand_kpts = person.get(hand_key, [])
            for i in range(0, len(hand_kpts), 3):
                if i + 1 < len(hand_kpts):
                    hand_kpts[i] *= scale_x
                    hand_kpts[i+1] *= scale_y
    
    # Update canvas dimensions
    normalized_pose[0]['canvas_width'] = target_canvas_width
    normalized_pose[0]['canvas_height'] = target_canvas_height
    
    return normalized_pose


def apply_canvas_size_scaling_to_coordinates(pose_keypoint, canvas_width, canvas_height):
        """
        Apply canvas size scaling to POSE_KEYPOINT coordinate values
        
        Args:
            pose_keypoint: POSE_KEYPOINT data (normalized 0-1 coordinates expected)
            canvas_width: Target canvas width for coordinate scaling
            canvas_height: Target canvas height for coordinate scaling
        
        Returns:
            pose_keypoint with coordinates scaled to canvas size
        """
        import copy
        
        if not pose_keypoint or len(pose_keypoint) == 0:
            return pose_keypoint
        
        scaled_pose = copy.deepcopy(pose_keypoint)
        
        for person in scaled_pose[0].get('people', []):
            # Scale body keypoints  
            body_kpts = person.get('pose_keypoints_2d', [])
            for i in range(0, len(body_kpts), 3):
                if i + 1 < len(body_kpts):
                    body_kpts[i] *= canvas_width        # x coordinate  
                    body_kpts[i+1] *= canvas_height     # y coordinate
                    # confidence stays the same
            
            # Scale face keypoints
            face_kpts = person.get('face_keypoints_2d', [])
            for i in range(0, len(face_kpts), 3):
                if i + 1 < len(face_kpts):
                    face_kpts[i] *= canvas_width
                    face_kpts[i+1] *= canvas_height
            
            # Scale hand keypoints
            for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                hand_kpts = person.get(hand_key, [])
                for i in range(0, len(hand_kpts), 3):
                    if i + 1 < len(hand_kpts):
                        hand_kpts[i] *= canvas_width
                        hand_kpts[i+1] *= canvas_height
        
        return scaled_pose


def draw_dwpose_render(pose_keypoint, resolution_x, show_body, show_face, show_hands, show_feet, 
                       pose_marker_size, face_marker_size, hand_marker_size):
    """
    Render POSE_KEYPOINT data using DWPose style with 25-point support including toe keypoints
    Compatible with ultimate-openpose-render parameters but using DWPose rendering algorithms
    """
    import cv2
    import math
    
    if not pose_keypoint or len(pose_keypoint) == 0:
        return []
    
    pose_imgs = []
    
    # Handle single frame vs multi-frame input
    if isinstance(pose_keypoint, dict):
        frames = [pose_keypoint]
    else:
        frames = pose_keypoint
    
    for frame_data in frames:
        if 'people' not in frame_data or len(frame_data['people']) == 0:
            # Create empty image for frames with no people
            H = frame_data.get('canvas_height', 768)
            W = frame_data.get('canvas_width', 512)
            if resolution_x > 0:
                W = resolution_x
                H = int(frame_data.get('canvas_height', 768) * (W / frame_data.get('canvas_width', 512)))
            pose_imgs.append(np.zeros((H, W, 3), dtype=np.uint8))
            continue
            
        # Get canvas dimensions
        H = frame_data.get('canvas_height', 768)
        W = frame_data.get('canvas_width', 512)
        
        # Apply resolution scaling
        if resolution_x > 0:
            W_scaled = resolution_x
            H_scaled = int(H * (W_scaled / W))
        else:
            W_scaled, H_scaled = W, H
        
        # Create canvas
        canvas = np.zeros((H_scaled, W_scaled, 3), dtype=np.uint8)
        
        # Process each person
        for person in frame_data['people']:
            if show_body:
                canvas = draw_body_dwpose_render(canvas, person, W, H, W_scaled, H_scaled, pose_marker_size, show_feet)
            
            if show_face:
                canvas = draw_face_dwpose_render(canvas, person, W, H, W_scaled, H_scaled, face_marker_size)
            
            if show_hands:
                canvas = draw_hands_dwpose_render(canvas, person, W, H, W_scaled, H_scaled, hand_marker_size)
        
        pose_imgs.append(canvas)
    
    return pose_imgs


def draw_body_dwpose_render(canvas, person, orig_w, orig_h, scaled_w, scaled_h, marker_size, show_feet):
    """
    Draw body keypoints with DWPose 25-point structure including toe keypoints
    """
    import cv2
    
    body_kpts = person.get('pose_keypoints_2d', [])
    if len(body_kpts) < 75:  # 25 points * 3
        return canvas
    
    # Scale factors
    scale_x = scaled_w / orig_w if orig_w > 0 else 1.0
    scale_y = scaled_h / orig_h if orig_h > 0 else 1.0
    
    # Extract keypoints
    points = []
    for i in range(25):
        x = body_kpts[i*3] * scale_x
        y = body_kpts[i*3+1] * scale_y
        conf = body_kpts[i*3+2]
        points.append([x, y, conf])
    
    # DWPose body connections (including toe connections)
    connections = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],  # Arms
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],  # Legs
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],  # Head
    ]
    
    # Add toe connections if show_feet is enabled
    if show_feet:
        connections.extend([[10, 18], [13, 19]])  # Toe connections
    
    # Draw connections
    for start_idx, end_idx in connections:
        if (start_idx < len(points) and end_idx < len(points) and
            points[start_idx][2] > 0.1 and points[end_idx][2] > 0.1):
            
            start_point = (int(points[start_idx][0]), int(points[start_idx][1]))
            end_point = (int(points[end_idx][0]), int(points[end_idx][1]))
            
            cv2.line(canvas, start_point, end_point, (0, 255, 0), 2)
    
    # Draw keypoints
    if marker_size > 0:
        for i, point in enumerate(points):
            if point[2] > 0.1:
                # Skip toe keypoints if show_feet is disabled
                if not show_feet and i in [18, 19]:
                    continue
                
                center = (int(point[0]), int(point[1]))
                cv2.circle(canvas, center, marker_size, (255, 0, 0), -1)
    
    return canvas


def draw_face_dwpose_render(canvas, person, orig_w, orig_h, scaled_w, scaled_h, marker_size):
    """
    Draw face keypoints
    """
    import cv2
    
    face_kpts = person.get('face_keypoints_2d', [])
    if len(face_kpts) < 210 or marker_size <= 0:  # 70 points * 3
        return canvas
    
    # Scale factors
    scale_x = scaled_w / orig_w if orig_w > 0 else 1.0
    scale_y = scaled_h / orig_h if orig_h > 0 else 1.0
    
    # Draw face keypoints
    for i in range(70):
        idx = i * 3
        if face_kpts[idx + 2] > 0.1:  # confidence check
            x = int(face_kpts[idx] * scale_x)
            y = int(face_kpts[idx + 1] * scale_y)
            cv2.circle(canvas, (x, y), marker_size, (255, 255, 0), -1)
    
    return canvas


def draw_hands_dwpose_render(canvas, person, orig_w, orig_h, scaled_w, scaled_h, marker_size):
    """
    Draw hand keypoints
    """
    import cv2
    
    if marker_size <= 0:
        return canvas
    
    # Scale factors
    scale_x = scaled_w / orig_w if orig_w > 0 else 1.0
    scale_y = scaled_h / orig_h if orig_h > 0 else 1.0
    
    # Draw left hand
    lhand_kpts = person.get('hand_left_keypoints_2d', [])
    if len(lhand_kpts) >= 63:  # 21 points * 3
        for i in range(21):
            idx = i * 3
            if lhand_kpts[idx + 2] > 0.1:
                x = int(lhand_kpts[idx] * scale_x)
                y = int(lhand_kpts[idx + 1] * scale_y)
                cv2.circle(canvas, (x, y), marker_size, (0, 0, 255), -1)
    
    # Draw right hand
    rhand_kpts = person.get('hand_right_keypoints_2d', [])
    if len(rhand_kpts) >= 63:  # 21 points * 3
        for i in range(21):
            idx = i * 3
            if rhand_kpts[idx + 2] > 0.1:
                x = int(rhand_kpts[idx] * scale_x)
                y = int(rhand_kpts[idx + 1] * scale_y)
                cv2.circle(canvas, (x, y), marker_size, (255, 0, 255), -1)
    
    return canvas


class ProportionChangerDWPoseRender:
    """
    DWPose Render Node with 25-point keypoint support including toe keypoints.
    Compatible with ultimate-openpose-render parameters but using DWPose rendering algorithms.
    Resolves coordinate misalignment issues when displaying ProportionChanger outputs.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True, "tooltip": "Draw body keypoints"}),
                "show_face": ("BOOLEAN", {"default": True, "tooltip": "Draw face keypoints"}),
                "show_hands": ("BOOLEAN", {"default": True, "tooltip": "Draw hand keypoints"}),
                "show_feet": ("BOOLEAN", {"default": True, "tooltip": "Draw toe keypoints (DWPose 25-point feature)"}),
                "resolution_x": ("INT", {"default": -1, "min": -1, "max": 12800, "tooltip": "Output width (-1 for original)"}),
                "pose_marker_size": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Body keypoint marker size"}),
                "face_marker_size": ("INT", {"default": 3, "min": 0, "max": 100, "tooltip": "Face keypoint marker size"}),
                "hand_marker_size": ("INT", {"default": 2, "min": 0, "max": 100, "tooltip": "Hand keypoint marker size"}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT", {"default": None, "tooltip": "POSE_KEYPOINT data to render"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_img"
    CATEGORY = "ProportionChanger"

    def render_img(self, show_body, show_face, show_hands, show_feet, resolution_x, 
                   pose_marker_size, face_marker_size, hand_marker_size, POSE_KEYPOINT=None):
        
        if POSE_KEYPOINT is None:
            raise ValueError("POSE_KEYPOINT input is required")
        
        # Debug logs disabled for legacy unianimate module (use PROPORTION_CHANGER_DEBUG=true to enable)
        # print(f"🔍 Debug POSE_KEYPOINT type: {type(POSE_KEYPOINT)}")
        # if isinstance(POSE_KEYPOINT, list) and len(POSE_KEYPOINT) > 0:
        #     print(f"🔍 Debug POSE_KEYPOINT length: {len(POSE_KEYPOINT)}")
        #     first_frame = POSE_KEYPOINT[0]
        #     print(f"🔍 Debug first frame keys: {first_frame.keys() if isinstance(first_frame, dict) else 'Not a dict'}")
        #     if isinstance(first_frame, dict) and 'people' in first_frame:
        #         print(f"🔍 Debug people count: {len(first_frame['people'])}")
        #         if len(first_frame['people']) > 0:
        #             person = first_frame['people'][0]
        #             if 'pose_keypoints_2d' in person:
        #                 keypoints = person['pose_keypoints_2d']
        #                 print(f"🔍 Debug pose_keypoints_2d length: {len(keypoints)}")
        #                 print(f"🔍 Debug first 9 keypoints: {keypoints[:9]}")
        #                 # Check for actual coordinate values
        #                 non_zero_coords = [(i//3, keypoints[i], keypoints[i+1], keypoints[i+2]) 
        #                                  for i in range(0, min(54, len(keypoints)), 3) 
        #                                  if keypoints[i] != 0 or keypoints[i+1] != 0]
        #                 print(f"🔍 Debug non-zero coordinates (first 5): {non_zero_coords[:5]}")
        # elif isinstance(POSE_KEYPOINT, dict):
        #     print(f"🔍 Debug single frame keys: {POSE_KEYPOINT.keys()}")
        
        # Render using DWPose algorithms
        pose_imgs = draw_dwpose_render(
            POSE_KEYPOINT, resolution_x, show_body, show_face, show_hands, show_feet,
            pose_marker_size, face_marker_size, hand_marker_size
        )
        
        if pose_imgs:
            # Convert to ComfyUI tensor format
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return (torch.from_numpy(pose_imgs_np),)
        else:
            raise ValueError("Invalid input type. Expected an input to give an output.")


NODE_CLASS_MAPPINGS = {
    "ProportionChangerDWPoseRender": ProportionChangerDWPoseRender,
    
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProportionChangerDWPoseRender": "ProportionChanger DWPose Render",
    
    }

    
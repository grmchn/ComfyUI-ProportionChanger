
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

class ProportionChangerUniAnimateDWPoseDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_images": ("IMAGE", {"tooltip": "Pose images"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose detection"}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "Stick width for drawing keypoints"}),
                "draw_body": ("BOOLEAN", {"default": True, "tooltip": "Draw body keypoints"}),
                "body_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Body keypoint size"}),
                "draw_feet": ("BOOLEAN", {"default": True, "tooltip": "Draw feet keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Draw hand keypoints"}),
                "hand_keypoint_size": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1, "tooltip": "Hand keypoint size"}),
                "colorspace": (["RGB", "BGR"], {"tooltip": "Color space for the output image"}),
                "handle_not_detected": (["empty", "repeat"], {"default": "empty", "tooltip": "How to handle undetected poses, empty inserts black and repeat inserts previous detection"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Draw head keypoints"}),
            },
            "optional": {
                "reference_pose_image": ("IMAGE", {"tooltip": "Reference pose image"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", )
    RETURN_NAMES = ("poses", "reference_pose",)
    FUNCTION = "process"
    CATEGORY = "ProportionChanger"

    def process(self, pose_images, score_threshold, stick_width, reference_pose_image=None, draw_body=True, body_keypoint_size=4, 
                draw_feet=True, draw_hands=True, hand_keypoint_size=4, colorspace="RGB", handle_not_detected="empty", draw_head=True):

        device = mm.get_torch_device()
        
        #model loading
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det=os.path.join(model_base_path, yolo_model)
        model_pose=os.path.join(model_base_path, dw_pose_model)

        if not os.path.exists(model_det):
            log.info(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            log.info(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det, map_location=device)
            self.pose = torch.jit.load(model_pose, map_location=device)
            self.dwpose_detector = DWposeDetector(self.det, self.pose) 

        #model inference
        height, width = pose_images.shape[1:3]
        
        pose_np = pose_images.cpu().numpy() * 255
        ref_np = None
        if reference_pose_image is not None:
            ref = reference_pose_image
            ref_np = ref.cpu().numpy() * 255

        poses, reference_pose = pose_extract(pose_np, ref_np, self.dwpose_detector, height, width, score_threshold, stick_width=stick_width,
                                             draw_body=draw_body, body_keypoint_size=body_keypoint_size, draw_feet=draw_feet, 
                                             draw_hands=draw_hands, hand_keypoint_size=hand_keypoint_size, handle_not_detected=handle_not_detected, draw_head=draw_head)
        poses = poses / 255.0
        if reference_pose_image is not None:
            reference_pose = reference_pose.unsqueeze(0) / 255.0
        else:
            reference_pose = torch.zeros(1, 64, 64, 3, device=torch.device("cpu"))

        if colorspace == "BGR":
            poses=torch.flip(poses, dims=[-1])

        return (poses, reference_pose, )


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


class ProportionChangerUltimateUniAnimateDWPoseDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "pose_keypoints": ("POSE_KEYPOINT", {"tooltip": "Target pose keypoints"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose processing"}),
            },
            "optional": {
                "reference_pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Reference pose keypoint"}),
            },
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("processed_pose_keypoint",)
    FUNCTION = "process"
    CATEGORY = "ProportionChanger"

    def process(self, pose_keypoints, score_threshold, reference_pose_keypoint=None):
        """
        Process POSE_KEYPOINT data using proportion changing algorithms
        """
        import numpy as np
        
        if not pose_keypoints or len(pose_keypoints) == 0:
            # Return empty keypoint data
            empty_person = {
                "pose_keypoints_2d": [0.0] * 75,
                "face_keypoints_2d": [0.0] * 210,
                "hand_left_keypoints_2d": [0.0] * 63,
                "hand_right_keypoints_2d": [0.0] * 63
            }
            return ([{"people": [empty_person], "canvas_width": 512, "canvas_height": 768}],)
        
        # Get canvas dimensions from first frame
        frame_data = pose_keypoints[0]
        canvas_width = frame_data.get('canvas_width', 512)
        canvas_height = frame_data.get('canvas_height', 768)
        
        # Convert POSE_KEYPOINT to DWPose format
        pose_data = pose_keypoint_to_dwpose_format(pose_keypoints, canvas_width, canvas_height)
        ref_data = None
        if reference_pose_keypoint is not None:
            ref_data = pose_keypoint_to_dwpose_format(reference_pose_keypoint, canvas_width, canvas_height)
        
        # Apply proportion changing algorithms (extracted from original code)
        processed_pose = self.apply_proportion_changes(pose_data, ref_data, score_threshold)
        
        # Convert back to POSE_KEYPOINT format
        result_keypoint = dwpose_format_to_pose_keypoint(
            processed_pose['bodies']['candidate'],
            processed_pose['faces'],
            processed_pose['hands'],
            canvas_width,
            canvas_height
        )
        
        return (result_keypoint,)
    
    def apply_proportion_changes(self, pose_data, ref_data, score_threshold):
        """
        Apply proportion changing algorithms from the original DWPose detector
        Extracted and adapted from pose_extract function (lines 273-600+)
        """
        import numpy as np
        
        # Get candidate and subset from pose data
        candidate = pose_data['bodies']['candidate']
        faces = pose_data['faces']
        hands = pose_data['hands']
        
        if len(candidate) == 0:
            return pose_data
        
        # If reference data is provided, apply proportion changes
        if ref_data is not None and len(ref_data['bodies']['candidate']) > 0:
            ref_candidate = ref_data['bodies']['candidate']
            ref_faces = ref_data['faces']
            ref_hands = ref_data['hands']
            
            # Apply the proportion changing algorithm (simplified version)
            # Based on shoulder and hip ratios from original code
            if len(candidate) >= 12 and len(ref_candidate) >= 12:
                # Calculate reference ratios
                ref_2_x, ref_2_y = ref_candidate[2][0], ref_candidate[2][1]  # RShoulder
                ref_5_x, ref_5_y = ref_candidate[5][0], ref_candidate[5][1]  # LShoulder
                ref_8_x, ref_8_y = ref_candidate[8][0], ref_candidate[8][1]  # RHip
                ref_11_x, ref_11_y = ref_candidate[11][0], ref_candidate[11][1]  # LHip
                
                ref_center1 = 0.5 * (ref_candidate[2] + ref_candidate[5])  # Shoulder center
                ref_center2 = 0.5 * (ref_candidate[8] + ref_candidate[11])  # Hip center
                
                # Calculate target ratios
                zero_2_x, zero_2_y = candidate[2][0], candidate[2][1]
                zero_5_x, zero_5_y = candidate[5][0], candidate[5][1]
                zero_8_x, zero_8_y = candidate[8][0], candidate[8][1]
                zero_11_x, zero_11_y = candidate[11][0], candidate[11][1]
                
                zero_center1 = 0.5 * (candidate[2] + candidate[5])
                zero_center2 = 0.5 * (candidate[8] + candidate[11])
                
                # Calculate scaling ratios
                if (zero_5_x - zero_2_x) != 0 and (zero_center2[1] - zero_center1[1]) != 0:
                    x_ratio = (ref_5_x - ref_2_x) / (zero_5_x - zero_2_x)
                    y_ratio = (ref_center2[1] - ref_center1[1]) / (zero_center2[1] - zero_center1[1])
                    
                    # Apply scaling to body keypoints
                    candidate[:, 0] *= x_ratio
                    candidate[:, 1] *= y_ratio
                    
                    # Apply scaling to faces and hands if present
                    if len(faces) > 0:
                        faces[:, :, 0] *= x_ratio
                        faces[:, :, 1] *= y_ratio
                    
                    if len(hands) > 0:
                        hands[:, :, 0] *= x_ratio
                        hands[:, :, 1] *= y_ratio
        
        return {
            'bodies': {
                'candidate': candidate,
                'subset': pose_data['bodies']['subset']
            },
            'faces': faces,
            'hands': hands
        }


class ProportionChangerDWPoseDetectorForPoseKeypoint:
    """
    DWPose detector node that extracts pose keypoints from image and outputs POSE_KEYPOINT format.
    This node is designed to work with ProportionChangerUltimateUniAnimateDWPoseDetector.
    Includes toe keypoints (19-24) which are essential for full pose estimation.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE", {"tooltip": "Input image for pose detection"}),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Score threshold for pose detection"}),
            }
        }

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoint",)
    FUNCTION = "detect_pose"
    CATEGORY = "ProportionChanger"

    def detect_pose(self, image, score_threshold):
        device = mm.get_torch_device()
        
        # Model loading
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        script_directory = os.path.dirname(os.path.abspath(__file__))
        model_base_path = os.path.join(script_directory, "models", "DWPose")

        model_det = os.path.join(model_base_path, yolo_model)
        model_pose = os.path.join(model_base_path, dw_pose_model)

        # Download models if not exists
        if not os.path.exists(model_det):
            log.info(f"Downloading yolo model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/yolox-onnx", 
                                allow_patterns=[f"*{yolo_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)
            
        if not os.path.exists(model_pose):
            log.info(f"Downloading dwpose model to: {model_base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="hr16/DWPose-TorchScript-BatchSize5", 
                                allow_patterns=[f"*{dw_pose_model}*"],
                                local_dir=model_base_path, 
                                local_dir_use_symlinks=False)

        # Initialize JIT models
        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det, map_location=device)
            self.pose = torch.jit.load(model_pose, map_location=device)
            self.dwpose_detector = DWposeDetector(self.det, self.pose) 

        # Process image using the same approach as working UniAnimate detector
        height, width = image.shape[1:3]
        image_np = image.cpu().numpy() * 255
        
        pose_keypoints = []
        comfy_pbar = ProgressBar(len(image_np))
        
        for i, img in enumerate(image_np):
            try:
                # Use the high-level DWPose detector call (same as working UniAnimate version)
                pose = self.dwpose_detector(img, score_threshold=score_threshold)
                
                # Convert to POSE_KEYPOINT format using actual canvas dimensions
                pose_keypoint_frame = dwpose_format_to_pose_keypoint(
                    pose['bodies']['candidate'], 
                    pose['faces'], 
                    pose['hands'], 
                    width,  # Use actual canvas width for pixel coordinates
                    height  # Use actual canvas height for pixel coordinates
                )
                
                # Add canvas size info for compatibility
                pose_keypoint_frame["canvas_width"] = width
                pose_keypoint_frame["canvas_height"] = height
                pose_keypoints.append(pose_keypoint_frame)
                
            except Exception as e:
                # Create empty pose data for failed detection
                empty_pose = {
                    "version": "1.0",
                    "people": [],
                    "canvas_width": width,
                    "canvas_height": height
                }
                pose_keypoints.append(empty_pose)
            
            comfy_pbar.update(1)

        return (pose_keypoints,)
    
    def _normalize_pose_coordinates(self, pose_keypoint, canvas_width, canvas_height):
        """
        Normalize pose coordinates from pixel values to 0-1 range
        """
        if not pose_keypoint or "people" not in pose_keypoint:
            return pose_keypoint
        
        for person in pose_keypoint["people"]:
            # Normalize body keypoints
            if "pose_keypoints_2d" in person:
                body_kpts = person["pose_keypoints_2d"]
                for i in range(0, len(body_kpts), 3):
                    if i+1 < len(body_kpts):
                        body_kpts[i] = body_kpts[i] / canvas_width      # x coordinate
                        body_kpts[i+1] = body_kpts[i+1] / canvas_height  # y coordinate
            
            # Normalize face keypoints
            if "face_keypoints_2d" in person:
                face_kpts = person["face_keypoints_2d"]
                for i in range(0, len(face_kpts), 3):
                    if i+1 < len(face_kpts):
                        face_kpts[i] = face_kpts[i] / canvas_width
                        face_kpts[i+1] = face_kpts[i+1] / canvas_height
            
            # Normalize hand keypoints
            for hand_key in ["hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                if hand_key in person:
                    hand_kpts = person[hand_key]
                    for i in range(0, len(hand_kpts), 3):
                        if i+1 < len(hand_kpts):
                            hand_kpts[i] = hand_kpts[i] / canvas_width
                            hand_kpts[i+1] = hand_kpts[i+1] / canvas_height
        
        return pose_keypoint


NODE_CLASS_MAPPINGS = {
    "ProportionChangerUniAnimateDWPoseDetector": ProportionChangerUniAnimateDWPoseDetector,
    "ProportionChangerUltimateUniAnimateDWPoseDetector": ProportionChangerUltimateUniAnimateDWPoseDetector,
    "ProportionChangerDWPoseDetectorForPoseKeypoint": ProportionChangerDWPoseDetectorForPoseKeypoint,
    
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProportionChangerUniAnimateDWPoseDetector": "ProportionChanger UniAnimate DWPose Detector",
    "ProportionChangerUltimateUniAnimateDWPoseDetector": "ProportionChanger Ultimate UniAnimate DWPose Detector",
    "ProportionChangerDWPoseDetectorForPoseKeypoint": "ProportionChanger DWPose Detector for POSE_KEYPOINT",
    
    }

    
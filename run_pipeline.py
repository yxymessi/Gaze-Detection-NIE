import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pdb
Image.MAX_IMAGE_PIXELS = None
# from scipy.misc import imresize
from PIL import Image
import cv2
import time

import argparse
import numpy as np
from PIL import Image
from FaceMaskDetection.utils.anchor_generator import generate_anchors
from FaceMaskDetection.utils.anchor_decode import decode_bbox
from FaceMaskDetection.utils.nms import single_class_non_max_suppression
from FaceMaskDetection.load_model.pytorch_loader import load_pytorch_model, pytorch_inference
from helper import update_frame_info_st, get_device_onwer, delete_none_student, mkdir, listdir
from scipy.ndimage import label
# from skimage.transform import resize as imresize

from attention_target_detection.model import ModelSpatial
from attention_target_detection.utils import imutils, evaluation
from attention_target_detection.config import *

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

# def _get_vistransform():
#     transform_list = []
#     transform_list.append(transforms.Resize((output_resolution, output_resolution)))
#     transform_list.append(transforms.ToTensor())
#     transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#     return transforms.Compose(transform_list)

def init_frame_info(frame_info):
    frame_info = {
        'st1': {'head_p': None, 'gaze_p': None, 'gaze_obj': None, 'gaze_hm': None},
        'st2': {'head_p': None, 'gaze_p': None, 'gaze_obj': None, 'gaze_hm': None},
        'st3': {'head_p': None, 'gaze_p': None, 'gaze_obj': None, 'gaze_hm': None},
        'obj': None
    }
    return frame_info


####################################################
# face detection
####################################################


def inference(image,
              conf_thresh=0.7,
              iou_thresh=0,
              target_shape=(160, 160),
              draw_result=False,
              show_result=False
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model_face, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        # output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        output_info.append([xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    if len(output_info) == 3:
        delete_none_student(output_info)
    elif len(output_info) > 3:
        print('WARNING: suport max 3 students.')
    return output_info

# task 2
def get_gaze(frame_raw, df1, idx, flg, facecolor=(0, 1, 0)):
    df1['left'] -= (df1['right'] - df1['left']) * 0.1
    df1['right'] += (df1['right'] - df1['left']) * 0.1
    df1['top'] -= (df1['bottom'] - df1['top']) * 0.1
    df1['bottom'] += (df1['bottom'] - df1['top']) * 0.1

    with torch.no_grad():
        width, height = frame_raw.size
        head_box = [df1['left'], df1['top'], df1['right'], df1['bottom']]
        head = frame_raw.crop((head_box))  # head crop
        head = test_transforms(head)  # transform inputs
        frame = test_transforms(frame_raw)
        head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

        head = head.unsqueeze(0).cuda()
        frame = frame.unsqueeze(0).cuda()
        head_channel = head_channel.unsqueeze(0).cuda()
        # forward pass
        raw_hm, _, inout = model_gaze(frame, head_channel, head)

        # heatmap modulation
        raw_hm = raw_hm.cpu().detach().numpy() * 255
        raw_hm = raw_hm.squeeze()
        inout = inout.cpu().detach().numpy()
        inout = 1 / (1 + np.exp(-inout))
        inout = (1 - inout) * 255
        norm_map = cv2.resize(raw_hm, (width, height)) - inout
        # vis
        norm_p = None
        if inout < args.out_threshold:  # in-frame gaze
            pred_x, pred_y = evaluation.argmax_pts(raw_hm)
            norm_p = [pred_x / output_resolution, pred_y / output_resolution]
        regions = find_cluster_regions(raw_hm,idx,flg)


        frame_np = np.array(frame_raw)
        raw_hm_resized = cv2.resize(raw_hm, (frame_np.shape[1], frame_np.shape[0])).astype(np.uint8)
        raw_hm_colored = cv2.applyColorMap(raw_hm_resized, cv2.COLORMAP_JET)
        raw_hm_inverted = cv2.bitwise_not(raw_hm_colored)
        combined = cv2.addWeighted(frame_np, 0.1, raw_hm_colored, 1.5, 0)
        # cv2.imshow(str(idx), combined)
        # out_dir = './data/acc_video/G3D5-Task4_out/'
        # cv2.imwrite(out_dir+str(idx)+'_'+flg+'.png', combined)
        # cv2.waitKey(0)

    return norm_p, head_box, inout, raw_hm ,regions




def find_cluster_regions(heatmap,idx,flg):
    num_clusters = 10
    threshold = 2 * np.mean(heatmap)

    heatmap = cv2.resize(heatmap, (224, 224))  #input resolution

    while num_clusters > 2 :
        heatmap[heatmap<= threshold] = 0
        labeled, num_clusters = label(heatmap)
        if num_clusters > 2:
            threshold += 0.1 * threshold  

    regions = []
    for cluster_label in range(1, num_clusters + 1):
        cluster_indices = np.where(labeled == cluster_label)
        region_points = np.column_stack((cluster_indices[1], cluster_indices[0]))  
        region_mean = np.mean(heatmap[cluster_indices])  
        regions.append({
            'region_points': region_points,
            'region_mean': region_mean
        })

    regions = sorted(regions, key=lambda x: x['region_mean'], reverse=True) 

    height, width = heatmap.shape
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i, region in enumerate(regions):
        region_points = region['region_points']
        color = tuple(np.random.randint(0, 256, 3).tolist())  
        for point in region_points:
            cv2.circle(blank_image, tuple(point), 1, color, -1)

    # out_dir = './data/acc_video/G6D1-Task3_out/'
    # cv2.imwrite(out_dir+str(idx)+'_'+flg+'_ano'+'.png', blank_image)
    # cv2.waitKey(0)
    # 归一化 成比例
    for i in range(len(regions)):
        regions[i]['region_points'] = regions[i]['region_points']/ 224
    return regions






def point_in_rectangle(point, rect_points):
    min_x = min(rect_points[0], rect_points[2])
    max_x = max(rect_points[0], rect_points[2])
    min_y = min(rect_points[1], rect_points[3])
    max_y = max(rect_points[1], rect_points[3])

    if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y:
        return True
    else:
        return False

def heatmap_in_rectangle(points, rect_points):
    min_x = min(rect_points[0], rect_points[2])
    max_x = max(rect_points[0], rect_points[2])
    min_y = min(rect_points[1], rect_points[3])
    max_y = max(rect_points[1], rect_points[3])
    results = []

    for point in points:
        if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y:
            results.append(True)
        else:
            results.append(False)
    return any(results)

# task 3 object detection
def get_gaze_obj(obj_results,frame_info):
    # global frame_info
    # classes = {63: 'laptop', 65: 'remote', 66: 'keyboard', 67: 'cell phone'}
    classes = {0: 'laptop', 1: 'tablet', 2: 'cell phone', 3: 'worksheet', 4: 'recording pen'}
    owner_classes = ['laptop', 'tablet']
    index_save = []
    box1, box2, box3, name1, name2, name3 = None, None, None, None, None, None
    obj_info = obj_results[0].boxes
    obj_id = obj_info.cls
    # check person 看对方脸
    # student1看2
    if frame_info['st1']['gaze_p'] is not None and frame_info['st2']['head_p'] is not None and\
            point_in_rectangle(
                frame_info['st1']['gaze_p'],
                [frame_info['st2']['head_p']['left'], frame_info['st2']['head_p']['top'],
                 frame_info['st2']['head_p']['right'], frame_info['st2']['head_p']['bottom']]
            ):
        box1 = [frame_info['st2']['head_p']['left'], frame_info['st2']['head_p']['top'], frame_info['st2']['head_p']['right'], frame_info['st2']['head_p']['bottom']]
        name1 = 'student2'
        
    # student1看3
    if frame_info['st1']['gaze_p'] is not None and frame_info['st3']['head_p'] is not None and\
            point_in_rectangle(
                frame_info['st1']['gaze_p'],
                [frame_info['st3']['head_p']['left'], frame_info['st3']['head_p']['top'],
                 frame_info['st3']['head_p']['right'], frame_info['st3']['head_p']['bottom']]
            ):
        box1 = [frame_info['st3']['head_p']['left'], frame_info['st3']['head_p']['top'], frame_info['st3']['head_p']['right'], frame_info['st3']['head_p']['bottom']]
        name1 = 'student3'
        
    # student2看1
    if frame_info['st2']['gaze_p'] is not None and frame_info['st1']['head_p'] is not None and\
            point_in_rectangle(frame_info['st2']['gaze_p'],
                               [frame_info['st1']['head_p']['left'], frame_info['st1']['head_p']['top'],
                                frame_info['st1']['head_p']['right'], frame_info['st1']['head_p']['bottom']]):
        box2 = [frame_info['st1']['head_p']['left'], frame_info['st1']['head_p']['top'],
                                frame_info['st1']['head_p']['right'], frame_info['st1']['head_p']['bottom']]
        name2 = 'student1'
        
    # student2看3
    if frame_info['st2']['gaze_p'] is not None and frame_info['st3']['head_p'] is not None and\
            point_in_rectangle(
                frame_info['st2']['gaze_p'],
                [frame_info['st3']['head_p']['left'], frame_info['st3']['head_p']['top'],
                 frame_info['st3']['head_p']['right'], frame_info['st3']['head_p']['bottom']]
            ):
        box2 = [frame_info['st3']['head_p']['left'], frame_info['st3']['head_p']['top'],
                                frame_info['st3']['head_p']['right'], frame_info['st3']['head_p']['bottom']]
        name2 = 'student3'
    
    # student3看1
    if frame_info['st3']['gaze_p'] is not None and frame_info['st1']['head_p'] is not None and\
            point_in_rectangle(
                frame_info['st3']['gaze_p'],
                [frame_info['st1']['head_p']['left'], frame_info['st1']['head_p']['top'],
                 frame_info['st1']['head_p']['right'], frame_info['st1']['head_p']['bottom']]
            ):
        box3 = [
            frame_info['st1']['head_p']['left'], frame_info['st1']['head_p']['top'],
            frame_info['st1']['head_p']['right'], frame_info['st1']['head_p']['bottom']
        ]
        name3 = 'student1'
    #
    # student3看2
    if frame_info['st3']['gaze_p'] is not None and frame_info['st2']['head_p'] is not None and\
            point_in_rectangle(
                frame_info['st3']['gaze_p'],
                [frame_info['st2']['head_p']['left'], frame_info['st2']['head_p']['top'],
                 frame_info['st2']['head_p']['right'], frame_info['st2']['head_p']['bottom']]
            ):
        box3 = [
            frame_info['st2']['head_p']['left'], frame_info['st2']['head_p']['top'],
            frame_info['st2']['head_p']['right'], frame_info['st2']['head_p']['bottom']
        ]
        name3 = 'student2'

    for i in range(len(obj_id)):
        if int(obj_info.cls[i]) in classes:
            # student box
            if frame_info['st1']['gaze_p'] is not None and \
                    point_in_rectangle(frame_info['st1']['gaze_p'], obj_info.xyxy[i]):
                box1 = obj_info.xyxy[i].cpu()
                class1 = classes[int(obj_info.cls[i])]
                if class1 in owner_classes:
                    onwer = get_device_onwer(frame_info, box1)
                    if onwer == 0:
                        onwer = 's'
                    else:
                        onwer = 'p'
                    name1 = f'{onwer} {class1}'
                else:
                    name1 = class1


            if frame_info['st2']['gaze_p'] is not None and \
                    point_in_rectangle(frame_info['st2']['gaze_p'], obj_info.xyxy[i]):
                box2 = obj_info.xyxy[i].cpu()
                class2 = classes[int(obj_info.cls[i])]
                if class2 in owner_classes:
                    onwer = get_device_onwer(frame_info, box2)
                    if onwer == 1:
                        onwer = 's'
                    else:
                        onwer = 'p'
                    name2 = f'{onwer} {class2}'
                else:
                    name2 = class2

            if frame_info['st3']['gaze_p'] is not None and \
                    point_in_rectangle(frame_info['st3']['gaze_p'], obj_info.xyxy[i]):
                box3 = obj_info.xyxy[i].cpu()
                class3 = classes[int(obj_info.cls[i])]
                if class3 in owner_classes:
                    onwer = get_device_onwer(frame_info, box3)
                    if onwer == 2:
                        onwer = 's'
                    else:
                        onwer = 'p'
                    name3 = f'{onwer} {class3}'
                else:
                    name3 = class3

    for i in range(len(obj_id)):
        if int(obj_info.cls[i]) in classes:
            if name1 is None and frame_info['st1']['gaze_hm']:
                if len(frame_info['st1']['gaze_hm']) > 1 and heatmap_in_rectangle(
                        torch.from_numpy(frame_info['st1']['gaze_hm'][0]['region_points']).cuda(),
                        [frame_info['st1']['head_p']['left'], frame_info['st1']['head_p']['top'],
                         frame_info['st1']['head_p']['right'], frame_info['st1']['head_p']['bottom']]):
                    if heatmap_in_rectangle(torch.from_numpy(frame_info['st1']['gaze_hm'][1]['region_points']).cuda(), obj_info.xyxy[i]):
                        box1 = obj_info.xyxy[i].cpu()
                        class1 = classes[int(obj_info.cls[i])]
                        if class1 in owner_classes:
                            onwer = get_device_onwer(frame_info, box1)
                            if onwer == 0:
                                onwer = 's'
                            else:
                                onwer = 'p'
                            name1 = f'{onwer} {class1}'
                        else:
                            name1 = class1

            if name1 is None and frame_info['st1']['gaze_hm']:
                if int(obj_info.cls[i]) == 0:
                    if heatmap_in_rectangle(torch.from_numpy(frame_info['st1']['gaze_hm'][0]['region_points']).cuda(), obj_info.xyxy[i]):
                        class1 = 'laptop'
                        box1 = obj_info.xyxy[i].cpu()
                        if class1 in owner_classes:
                            onwer = get_device_onwer(frame_info, box1)
                            if onwer == 0:
                                onwer = 's'
                            else:
                                onwer = 'p'
                            name1 = f'{onwer} {class1}'
                        else:
                            name1 = class1

            if name2 is None and frame_info['st2']['gaze_hm']:
                if len(frame_info['st2']['gaze_hm']) > 1 and heatmap_in_rectangle(
                        torch.from_numpy(frame_info['st2']['gaze_hm'][0]['region_points']).cuda(),
                        [frame_info['st2']['head_p']['left'], frame_info['st2']['head_p']['top'],
                         frame_info['st2']['head_p']['right'], frame_info['st2']['head_p']['bottom']]):
                    if heatmap_in_rectangle(torch.from_numpy(frame_info['st2']['gaze_hm'][1]['region_points']).cuda(), obj_info.xyxy[i]):
                        box2 = obj_info.xyxy[i].cpu()
                        class2 = classes[int(obj_info.cls[i])]
                        if class2 in owner_classes:
                            onwer = get_device_onwer(frame_info, box2)
                            if onwer == 1:
                                onwer = 's'
                            else:
                                onwer = 'p'
                            name2 = f'{onwer} {class2}'
                        else:
                            name2 = class2

            if name2 is None and frame_info['st2']['gaze_hm']:
                if int(obj_info.cls[i]) == 0:
                    if heatmap_in_rectangle(torch.from_numpy(frame_info['st2']['gaze_hm'][0]['region_points']).cuda(), obj_info.xyxy[i]):
                        class2 = 'laptop'
                        box2 = obj_info.xyxy[i].cpu()
                        if class2 in owner_classes:
                            onwer = get_device_onwer(frame_info, box2)
                            if onwer == 1:
                                onwer = 's'
                            else:
                                onwer = 'p'
                            name2 = f'{onwer} {class2}'
                        else:
                            name2 = class2

            if name3 is None and frame_info['st3']['gaze_hm']:
                if len(frame_info['st3']['gaze_hm']) > 1 and heatmap_in_rectangle(
                        torch.from_numpy(frame_info['st3']['gaze_hm'][0]['region_points']).cuda(),
                        [frame_info['st3']['head_p']['left'], frame_info['st3']['head_p']['top'],
                         frame_info['st3']['head_p']['right'], frame_info['st3']['head_p']['bottom']]):
                    if heatmap_in_rectangle(torch.from_numpy(frame_info['st3']['gaze_hm'][1]['region_points']).cuda(),
                                            obj_info.xyxy[i]):
                        box3 = obj_info.xyxy[i].cpu()
                        class3 = classes[int(obj_info.cls[i])]
                        if class3 in owner_classes:
                            onwer = get_device_onwer(frame_info, box3)
                            if onwer == 0:
                                onwer = 's'
                            else:
                                onwer = 'p'
                            name3 = f'{onwer} {class3}'
                        else:
                            name3 = class3

            if name3 is None and frame_info['st3']['gaze_hm']:
                if int(obj_info.cls[i]) == 0:
                    if heatmap_in_rectangle(torch.from_numpy(frame_info['st3']['gaze_hm'][0]['region_points']).cuda(),
                                            obj_info.xyxy[i]):
                        name3 = 'laptop'
                        box3 = obj_info.xyxy[i].cpu()



    return box1, name1, box2, name2, box3, name3
    # return box1, name1, box2, name2

# 计算iou
def compute_iou(headbox1, headbox2):
    """
    参数：
    headbox1 第一个人的位置数据
    headbox2 第二个人的位置数据
    return：iou值, 大头框， 小头框
    """
    box1_area = (headbox1[2] - headbox1[0]) * (headbox1[3] - headbox1[1])
    box2_area = (headbox2[2] - headbox2[0]) * (headbox2[3] - headbox2[1])
    sum_area = box1_area + box2_area
    left = max(headbox1[0], headbox2[0])
    right = min(headbox1[2], headbox2[2])
    bottom = max(headbox1[1], headbox2[1])
    top = min(headbox1[3], headbox2[3])
    if left >= right or bottom >= top:
        return 0, 0, 0
    else:
        inter = (right - left) * (top - bottom)
        iou = (inter / (sum_area - inter)) * 1.0
        if box1_area >= box2_area:
            return iou, 1.0, 2.0
        else:
            return iou, 2.0, 1.0


def run_on_video(video_path, output_video_name, conf_thresh):

    # preparation
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    csv_path = video_path+'.csv'
    raw_img_dir_path = '.' + video_path.split('.')[1]
    mkdir(raw_img_dir_path)
    out_dir_path = '.' + video_path.split('.')[1]+'_out'
    mkdir(out_dir_path)


    with open(csv_path, 'w+') as csvF:
        while status:
            start_stamp = time.time()
            status, img_raw = cap.read()
            if status and (idx % args.frame == 0):  # 10
                cv2.imwrite(f'{raw_img_dir_path}/{idx}.png', img_raw)
            # if not status:
            #     break
            try:
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            except:
                break
            read_frame_stamp = time.time()
            idx += 1
            if status and (idx % args.frame == 0) and (idx > 500):  # 10
                # get face info
                frame_info = {}
                frame_info = init_frame_info(frame_info)
                face_posi = inference(img_raw,
                                      conf_thresh=conf_thresh,
                                      iou_thresh=args.face_iou,
                                      target_shape=(360, 360),
                                      draw_result=False,
                                      show_result=False)

                # face_posi: [[l, b, r, t], []]
                face_state = update_frame_info_st(face_posi, frame_info)

                # TODO: update helper.py-add one face and more than 2 faces
                if face_state is not None:
                    frame_info = face_state
                    frame_raw = Image.fromarray(np.uint8(img_raw))
                    obj_results = model_obj(frame_raw)

                    # student change
                    if frame_info['st1']['head_p'] is not None:
                        norm_p1, head_box1, inout1, raw_hm1,regions_1 = get_gaze(frame_raw, frame_info['st1']['head_p'],idx,flg = 'st1',facecolor=(1, 0, 0))
                        for i in range(len(regions_1)):
                            regions_1[i]['region_points'] = regions_1[i]['region_points'] * np.array([width, height])
                        frame_info['st1']['gaze_hm'] = regions_1
                        if norm_p1 is not None:
                            frame_info['st1']['gaze_p'] = [norm_p1[0] * width, norm_p1[1] * height]
                        else:
                            frame_info['st1']['gaze_p'] = None

                    if frame_info['st2']['head_p'] is not None:
                        norm_p2, head_box2, inout2, raw_hm2, regions_2 = get_gaze(frame_raw, frame_info['st2']['head_p'],
                                                                       idx, flg = 'st2', facecolor=(1, 0, 0))
                        for i in range(len(regions_2)):
                            regions_2[i]['region_points'] = regions_2[i]['region_points'] * np.array([width, height])
                        frame_info['st2']['gaze_hm'] = regions_2

                        if norm_p2 is not None:
                            frame_info['st2']['gaze_p'] = [norm_p2[0] * width, norm_p2[1] * height]
                        else:
                            frame_info['st2']['gaze_p'] = None

                    if frame_info['st3']['head_p'] is not None and args.detect_num != 2:
                        norm_p3, head_box3, inout3, raw_hm3, regions_3 = get_gaze(frame_raw, frame_info['st3']['head_p'],
                                                                       idx, flg = 'st3' ,facecolor=(0, 1, 0))
                        frame_info['st3']['gaze_hm'] = regions_3
                        if norm_p3 is not None:
                            frame_info['st3']['gaze_p'] = [norm_p3[0] * width, norm_p3[1] * height]
                        else:
                            frame_info['st3']['gaze_p'] = None

                    # task 3
                    box1, name1, box2, name2, box3, name3 = get_gaze_obj(obj_results,frame_info)

                    # print('st1', name1, 'st2', name2, 'st3', name3)

                    # draw
                    plt.close()
                    fig = plt.figure()
                    # fig.canvas.manager.window.move(0, 0)
                    plt.axis('off')
                    plt.imshow(frame_raw)
                    ax = plt.gca()

                    # 新增iou逻辑判断
                    if frame_info['st1']['gaze_p'] is not None:
                        rect1 = patches.Rectangle((head_box1[0], head_box1[1]), head_box1[2] - head_box1[0],
                                                  head_box1[3] - head_box1[1], linewidth=1, edgecolor=(0, 1, 0, 0.5),
                                                  facecolor='none')
                        ax.add_patch(rect1)

                        if box1 is not None:
                            circ1 = patches.Circle(((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2), height / 50.0,
                                                   facecolor=(0, 1, 0),
                                                   edgecolor='none')
                        else:
                            circ1 = patches.Circle((norm_p1[0] * width, norm_p1[1] * height), height / 50.0,
                                               facecolor=(0, 1, 0),
                                               edgecolor='none')
                        ax.add_patch(circ1)
                        if box1 is not None:
                            plt.plot(((box1[0] + box1[2]) / 2, (head_box1[0] + head_box1[2]) / 2),
                                     ((box1[1] + box1[3]) / 2, (head_box1[1] + head_box1[3]) / 2), '-',
                                     color=(0, 1, 0, 1))
                        else:
                            plt.plot((norm_p1[0] * width, (head_box1[0] + head_box1[2]) / 2),
                                    (norm_p1[1] * height, (head_box1[1] + head_box1[3]) / 2), '-', color=(0, 1, 0, 1))
                        if box1 is not None:
                            rect_obj1 = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0],
                                                          box1[3] - box1[1], linewidth=2, edgecolor=(0, 0, 1),
                                                          facecolor='none')
                            ax.add_patch(rect_obj1)
                    if frame_info['st2']['gaze_p'] is not None:
                        rect2 = patches.Rectangle((head_box2[0], head_box2[1]), head_box2[2] - head_box2[0],
                                                  head_box2[3] - head_box2[1], linewidth=1, edgecolor=(0, 1, 0, 0.5),
                                                  facecolor='none')
                        ax.add_patch(rect2)
                        if box2 is not None:
                            circ2 = patches.Circle(((box2[0]+box2[2])/2, (box2[1]+box2[3])/2), height / 50.0,
                                                   facecolor=(0, 1, 0),
                                                   edgecolor='none')
                        else:
                            circ2 = patches.Circle((norm_p2[0] * width, norm_p2[1] * height), height / 50.0,
                                               facecolor=(0, 1, 0),
                                               edgecolor='none')
                        ax.add_patch(circ2)
                        if box2 is not None:
                            plt.plot(((box2[0]+box2[2])/2, (head_box2[0] + head_box2[2]) / 2),
                                    ((box2[1]+box2[3])/2, (head_box2[1] + head_box2[3]) / 2), '-', color=(0, 1, 0, 1))
                        else:
                            plt.plot((norm_p2[0] * width, (head_box2[0] + head_box2[2]) / 2),
                                     (norm_p2[1] * height, (head_box2[1] + head_box2[3]) / 2), '-', color=(0, 1, 0, 1))
                        if box2 is not None:
                            rect_obj2 = patches.Rectangle((box2[0], box2[1]), box2[2] - box2[0],
                                                          box2[3] - box2[1], linewidth=2, edgecolor=(1, 0, 0),
                                                          facecolor='none')
                            ax.add_patch(rect_obj2)
                    if frame_info['st3']['gaze_p'] is not None:
                        rect3 = patches.Rectangle((head_box3[0], head_box3[1]), head_box3[2] - head_box3[0],
                                                  head_box3[3] - head_box3[1], linewidth=1, edgecolor=(0, 1, 0, 0.5),
                                                  facecolor='none')
                        ax.add_patch(rect3)
                        circ3 = patches.Circle((norm_p3[0] * width, norm_p3[1] * height), height / 50.0,
                                               facecolor=(0, 1, 0),
                                               edgecolor='none')
                        ax.add_patch(circ3)
                        plt.plot((norm_p3[0] * width, (head_box3[0] + head_box3[2]) / 2),
                                 (norm_p3[1] * height, (head_box3[1] + head_box3[3]) / 2), '-', color=(0, 1, 0, 1))
                        if box3 is not None:
                            rect_obj3 = patches.Rectangle((box3[0], box3[1]), box3[2] - box3[0],
                                                          box3[3] - box3[1], linewidth=2, edgecolor=(1, 0, 0),
                                                          facecolor='none')
                            ax.add_patch(rect_obj3)

                    # plt.show(block=False)
                    plt.savefig(out_dir_path+f'/{idx}.png')
                    # pdb.set_trace()
                    plt.pause(0.2)
                else:
                    continue
                # cv2.imshow('image', img_raw[:, :, ::-1])
                # cv2.waitKey(1)
                inference_stamp = time.time()
                # writer.write(img_raw)
                write_frame_stamp = time.time()

                # print("%d of %d" % (idx, total_frames))
                csvF.write(f'{idx}/{total_frames}, {name1}, {name2}, {name3}\n')
                # print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                #                                                        inference_stamp - read_frame_stamp,
                #                                                        write_frame_stamp - inference_stamp))



    # writer.release()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', type=str, help='model weights', default='./attention_target_detection/chp/model_demo.pt')
    parser.add_argument('--image_dir', type=str, help='images', default='data/demo/frames')
    parser.add_argument('--video_path', type=str, help='video path', default='./attention_target_detection/data/video1.avi')
    parser.add_argument('--head', type=str, help='head bounding boxes', default='./attention_target_detection/data/demo/person2.txt')
    parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')  # heatmap
    parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
    parser.add_argument('--frame', type=int, help='detect one in num of frames ', default=10)  # 10
    parser.add_argument('--detect_num', type=int, help='detect the number of valid person', default=3)
    parser.add_argument('--face_conf', type=float, help='face position conf', default=0.8)
    parser.add_argument('--face_iou', type=float, help='face position conf', default=0.3)
    parser.add_argument('--obj_conf', type=float, help='face position conf', default=0.4)

    args = parser.parse_args()
    
    model_face = load_pytorch_model(r'./FaceMaskDetection/models/model360.pth')
    # anchor configuration
    #feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5

    # generate anchors
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)

    id2class = {0: 'Mask', 1: 'NoMask'}
    frame_info = {
            'st1': {'head_p': None, 'gaze_p': None, 'gaze_obj': None, 'gaze_hm': None},
            'st2': {'head_p': None, 'gaze_p': None, 'gaze_obj': None, 'gaze_hm': None},
            'st3': {'head_p': None, 'gaze_p': None, 'gaze_obj': None, 'gaze_hm': None},
            'obj': None
        }


    # model load for gaze
    # set up data transformation
    test_transforms = _get_transform()
    model_gaze = ModelSpatial()
    model_dict = model_gaze.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model_gaze.load_state_dict(model_dict)
    model_gaze.cuda()
    model_gaze.train(False)
    ######################
    from ultralytics import YOLO
    # model_obj = YOLO(r".\Yolov5_DeepSort_Pytorch\yolov8\ultralytics\models\v8\yolov8x_stu.yaml")
    model_obj = YOLO(r'./Yolov5_DeepSort_Pytorch/best.pt')


    run_on_video('./data/acc_video/G6D4-Task4.mov', '', conf_thresh=args.face_conf)

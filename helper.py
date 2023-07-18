import numpy as np
import os


def rank_pos(face_posi):
    '''
    返回从左到右的人头排序
    '''
    n = len(face_posi)
    templist = []

    # 去除干扰人脸信息
    for i in range(n):
        templist.append(face_posi[i][1])
    if len(templist) != 0:
        min_top = min(templist)
        if min_top == 0:
            min_top = 1e-8
    else:
        min_top = 1000000.0
    # print('min:', min_top)
    # print('templist',templist)
    for person_top in templist:
        if round(person_top / min_top) >= 10:
            if min_top == 1e-8:
                del face_posi[templist.index(0)]
            else:
                del face_posi[templist.index(min_top)]
    # print('after:', face_posi)
    # 将人头排序，从左到右
    if len(face_posi) == 2:
        if face_posi[0][0] < face_posi[1][0]:
            return face_posi
        else:
            return [face_posi[1], face_posi[0]]
    elif len(face_posi) == 3:
        if face_posi[0][0] < face_posi[1][0]:
            if face_posi[1][0] < face_posi[2][0]:
                return face_posi
            else:
                if face_posi[0][0] < face_posi[2][0]:
                    return [face_posi[0], face_posi[2], face_posi[1]]
                else:
                    return [face_posi[2], face_posi[0], face_posi[1]]
        else:
            if face_posi[0][0] < face_posi[2][0]:
                return [face_posi[1], face_posi[0], face_posi[2]]
            else:
                if face_posi[1][0] < face_posi[2][0]:
                    return [face_posi[1], face_posi[2], face_posi[0]]
                else:
                    return [face_posi[2], face_posi[1], face_posi[0]]
    else:  # TODO: 添加一人情况
        return face_posi


def update_frame_info_st(face_posi, frame_info):
    '''
    face_posi: face position [[l, t, r, b], []...]
    frame_info = {
        'st1': {'head_p': None, 'gaze_p': None, 'gaze_obj': None},
        'st2': {'head_p': None, 'gaze_p': None, 'gaze_obj': None},
        'st3': {'head_p': None, 'gaze_p': None, 'gaze_obj': None},
        'obj': None
    }
    '''
    # TODO: 处理只检测一个人脸情况，多个人脸情况
    # 新增逻辑：增加去除外来第三者，单人检测，多人检测
    if len(face_posi) == 0:
        return None
    elif len(face_posi) == 1:
        frame_info['st1']['head_p'] = {
            'left': face_posi[0][0],
            'top': face_posi[0][1],
            'right': face_posi[0][2],
            'bottom': face_posi[0][3]
        }
    else:
        face_posi = rank_pos(face_posi)
        # print('leftafter: ', face_posi)
        for index in range(len(face_posi)):
            frame_info['st{}'.format(index + 1)]['head_p'] = {
                    'left': face_posi[index][0],
                    'top': face_posi[index][1],
                    'right': face_posi[index][2],
                    'bottom': face_posi[index][3]
            }
            # print('frame_info:', frame_info)
        return frame_info

def get_device_onwer(frame_info, box):
    st1_box, st2_box, st3_box = None, None, None
    if frame_info['st1']['head_p'] is not None:
        st1_box = [frame_info['st1']['head_p']['left'], frame_info['st1']['head_p']['top'], frame_info['st1']['head_p']['right'],
            frame_info['st1']['head_p']['bottom']]
    st1_p = get_center(st1_box)

    if frame_info['st2']['head_p'] is not None:
        st2_box = [frame_info['st2']['head_p']['left'], frame_info['st2']['head_p']['top'], frame_info['st2']['head_p']['right'],
            frame_info['st2']['head_p']['bottom']]
    st2_p = get_center(st2_box)

    if frame_info['st3']['head_p'] is not None:
        st3_box = [frame_info['st3']['head_p']['left'], frame_info['st3']['head_p']['top'], frame_info['st3']['head_p']['right'],
            frame_info['st3']['head_p']['bottom']]
    st3_p = get_center(st3_box)
    box_p = get_center(box)
    list_p = [10000, 10000, 10000]
    if st1_p is not None:
        list_p[0] = np.linalg.norm(np.array(st1_p) - np.array(box_p))
    if st2_p is not None:
        list_p[1] = np.linalg.norm(np.array(st2_p) - np.array(box_p))
    if st3_p is not None:
        list_p[2] = np.linalg.norm(np.array(st3_p) - np.array(box_p))
    min_index = np.argmin(list_p)
    return min_index

def get_center(box_list):
    if box_list is not None:
        return [(box_list[0]+box_list[2])/2, (box_list[1]+box_list[3])/2]
    else:
        return None

def delete_none_student(output_info):
    y_list = [
        (output_info[0][1] + output_info[0][3]) / 2,
        (output_info[1][1] + output_info[1][3]) / 2,
        (output_info[2][1] + output_info[2][3]) / 2,
    ]
    y_index = np.argmin(np.array(y_list))
    del output_info[y_index]


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
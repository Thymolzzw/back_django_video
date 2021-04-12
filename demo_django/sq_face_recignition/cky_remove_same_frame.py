import os
import uuid

import cv2
import numpy as np
from skimage.io import imread
from demo_django.sq_face_recignition.facerecoginition_knn import *
import time


def extract_image_return_numpy(path, num):  # 抗裁剪解水印时调用，将原始视频抽帧
    cap = cv2.VideoCapture(path)  # 读入文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)  # 从num帧开始读视频
    success, frame = cap.read()
    # print("frame", frame)
    cap.release()
    return success, frame

def read(image):
    # Step 1:将图像加载为灰度数组
    im_array = imread(image, as_gray=True)

    # Step 2a:确定裁剪边界
    rw = np.cumsum(np.sum(np.abs(np.diff(im_array, axis=1)), axis=1))
    cw = np.cumsum(np.sum(np.abs(np.diff(im_array, axis=0)), axis=0))
    upper_column_limit = np.searchsorted(cw, np.percentile(cw, 95), side='left')
    lower_column_limit = np.searchsorted(cw, np.percentile(cw, 5), side='right')
    upper_row_limit = np.searchsorted(rw, np.percentile(rw, 95), side='left')
    lower_row_limit = np.searchsorted(rw, np.percentile(rw, 5), side='right')
    if lower_row_limit > upper_row_limit:
        lower_row_limit = int(5 / 100. * im_array.shape[0])
        upper_row_limit = int(95 / 100. * im_array.shape[0])
    if lower_column_limit > upper_column_limit:
        lower_column_limit = int(5 / 100. * im_array.shape[1])
        upper_column_limit = int(95 / 100. * im_array.shape[1])
    image_limits = [(lower_row_limit, upper_row_limit), (lower_column_limit, upper_column_limit)]

    # Step 2b:生成网格中心
    x_coords = np.linspace(image_limits[0][0], image_limits[0][1], 11, dtype=int)[1:-1]
    y_coords = np.linspace(image_limits[1][0], image_limits[1][1], 11, dtype=int)[1:-1]

    # Step 3:计算以每个网格点为中心的每个P x P平方的灰度平均值
    P = max([2.0, int(0.5 + min(im_array.shape) / 20.)])
    avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))
    for i, x in enumerate(x_coords):
        lower_x_lim = int(max([x - P / 2, 0]))
        upper_x_lim = int(min([lower_x_lim + P, im_array.shape[0]]))
        for j, y in enumerate(y_coords):
            lower_y_lim = int(max([y - P / 2, 0]))
            upper_y_lim = int(min([lower_y_lim + P, im_array.shape[1]]))
            avg_grey[i, j] = np.mean(im_array[lower_x_lim:upper_x_lim,lower_y_lim:upper_y_lim])

    # Step 4a:计算每个网格点相对于每个邻居的差异数组
    right_neighbors = -np.concatenate((np.diff(avg_grey), np.zeros(avg_grey.shape[0]).reshape((avg_grey.shape[0], 1))), axis=1)
    left_neighbors = -np.concatenate((right_neighbors[:, -1:], right_neighbors[:, :-1]), axis=1)
    down_neighbors = -np.concatenate((np.diff(avg_grey, axis=0),np.zeros(avg_grey.shape[1]).reshape((1, avg_grey.shape[1]))))
    up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))
    diagonals = np.arange(-avg_grey.shape[0] + 1, avg_grey.shape[0])
    upper_left_neighbors = sum([np.diagflat(np.insert(np.diff(np.diag(avg_grey, i)), 0, 0), i) for i in diagonals])
    lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:], (0, 1), mode='constant')
    flipped = np.fliplr(avg_grey)
    upper_right_neighbors = sum([np.diagflat(np.insert(np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
    lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:], (0, 1), mode='constant')
    diff_mat = np.dstack(np.array([upper_left_neighbors, up_neighbors, np.fliplr(upper_right_neighbors), left_neighbors, right_neighbors,np.fliplr(lower_left_neighbors), down_neighbors, lower_right_neighbors]))

    # Step 4b:舍弃差异仅为2n+1的值
    mask = np.abs(diff_mat) < 2 / 255.
    diff_mat[mask] = 0.
    positive_cutoffs = np.percentile(diff_mat[diff_mat > 0.], np.linspace(0, 100, 3))
    negative_cutoffs = np.percentile(diff_mat[diff_mat < 0.], np.linspace(100, 0, 3))
    for level, interval in enumerate([positive_cutoffs[i:i + 2] for i in range(positive_cutoffs.shape[0] - 1)]):
        diff_mat[(diff_mat >= interval[0]) & (diff_mat <= interval[1])] = level + 1
    for level, interval in enumerate([negative_cutoffs[i:i + 2] for i in range(negative_cutoffs.shape[0] - 1)]):
        diff_mat[(diff_mat <= interval[0]) & (diff_mat >= interval[1])] = -(level + 1)

    # Step 5:展平数组并返回特征
    return np.ravel(diff_mat).astype('int8')



def read_numpy(image):
    # Step 1:将图像加载为灰度数组
    # im_array = imread(image, as_gray=True)

    im_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2a:确定裁剪边界
    rw = np.cumsum(np.sum(np.abs(np.diff(im_array, axis=1)), axis=1))
    cw = np.cumsum(np.sum(np.abs(np.diff(im_array, axis=0)), axis=0))
    upper_column_limit = np.searchsorted(cw, np.percentile(cw, 95), side='left')
    lower_column_limit = np.searchsorted(cw, np.percentile(cw, 5), side='right')
    upper_row_limit = np.searchsorted(rw, np.percentile(rw, 95), side='left')
    lower_row_limit = np.searchsorted(rw, np.percentile(rw, 5), side='right')
    if lower_row_limit > upper_row_limit:
        lower_row_limit = int(5 / 100. * im_array.shape[0])
        upper_row_limit = int(95 / 100. * im_array.shape[0])
    if lower_column_limit > upper_column_limit:
        lower_column_limit = int(5 / 100. * im_array.shape[1])
        upper_column_limit = int(95 / 100. * im_array.shape[1])
    image_limits = [(lower_row_limit, upper_row_limit), (lower_column_limit, upper_column_limit)]

    # Step 2b:生成网格中心
    x_coords = np.linspace(image_limits[0][0], image_limits[0][1], 11, dtype=int)[1:-1]
    y_coords = np.linspace(image_limits[1][0], image_limits[1][1], 11, dtype=int)[1:-1]

    # Step 3:计算以每个网格点为中心的每个P x P平方的灰度平均值
    P = max([2.0, int(0.5 + min(im_array.shape) / 20.)])
    avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))
    for i, x in enumerate(x_coords):
        lower_x_lim = int(max([x - P / 2, 0]))
        upper_x_lim = int(min([lower_x_lim + P, im_array.shape[0]]))
        for j, y in enumerate(y_coords):
            lower_y_lim = int(max([y - P / 2, 0]))
            upper_y_lim = int(min([lower_y_lim + P, im_array.shape[1]]))
            avg_grey[i, j] = np.mean(im_array[lower_x_lim:upper_x_lim,lower_y_lim:upper_y_lim])

    # Step 4a:计算每个网格点相对于每个邻居的差异数组
    right_neighbors = -np.concatenate((np.diff(avg_grey), np.zeros(avg_grey.shape[0]).reshape((avg_grey.shape[0], 1))), axis=1)
    left_neighbors = -np.concatenate((right_neighbors[:, -1:], right_neighbors[:, :-1]), axis=1)
    down_neighbors = -np.concatenate((np.diff(avg_grey, axis=0),np.zeros(avg_grey.shape[1]).reshape((1, avg_grey.shape[1]))))
    up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))
    diagonals = np.arange(-avg_grey.shape[0] + 1, avg_grey.shape[0])
    upper_left_neighbors = sum([np.diagflat(np.insert(np.diff(np.diag(avg_grey, i)), 0, 0), i) for i in diagonals])
    lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:], (0, 1), mode='constant')
    flipped = np.fliplr(avg_grey)
    upper_right_neighbors = sum([np.diagflat(np.insert(np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
    lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:], (0, 1), mode='constant')
    diff_mat = np.dstack(np.array([upper_left_neighbors, up_neighbors, np.fliplr(upper_right_neighbors), left_neighbors, right_neighbors,np.fliplr(lower_left_neighbors), down_neighbors, lower_right_neighbors]))

    # Step 4b:舍弃差异仅为2n+1的值
    mask = np.abs(diff_mat) < 2 / 255.
    diff_mat[mask] = 0.
    positive_cutoffs = np.percentile(diff_mat[diff_mat > 0.], np.linspace(0, 100, 3))
    negative_cutoffs = np.percentile(diff_mat[diff_mat < 0.], np.linspace(100, 0, 3))
    for level, interval in enumerate([positive_cutoffs[i:i + 2] for i in range(positive_cutoffs.shape[0] - 1)]):
        diff_mat[(diff_mat >= interval[0]) & (diff_mat <= interval[1])] = level + 1
    for level, interval in enumerate([negative_cutoffs[i:i + 2] for i in range(negative_cutoffs.shape[0] - 1)]):
        diff_mat[(diff_mat <= interval[0]) & (diff_mat >= interval[1])] = -(level + 1)

    # Step 5:展平数组并返回特征
    return np.ravel(diff_mat).astype('int8')

def distance(image1, image2):
    image1 = read(image1)
    image2 = read(image2)
    norm_diff = np.linalg.norm(image1 - image2)
    norm1 = np.linalg.norm(image1)
    norm2 = np.linalg.norm(image2)
    return norm_diff / (norm1 + norm2)

def distance_with_numpy(image1, image2):
    image1 = read_numpy(image1)
    image2 = read_numpy(image2)
    norm_diff = np.linalg.norm(image1 - image2)
    norm1 = np.linalg.norm(image1)
    norm2 = np.linalg.norm(image2)
    return norm_diff / (norm1 + norm2)

def main(video_file_path, file):
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"

    capture = cv2.VideoCapture(video_file_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("frame_count:", frame_count, " fps:", fps)
    frame = 0.00
    frames = []
    list_knownface = []
    while int(frame + 0.5) < frame_count:
        # 先抽出来
        success, now_frame = extract_image_return_numpy(video_file_path, int(frame + 0.5))
        # print("now_frame", now_frame)
        if success == False:
            frame += fps
            print("end")
            continue

        # 再比对特征
        if len(frames) < 1:

            frame += fps
            frames.append(now_frame)
            continue
        ex_frame = frames[-1]
        dis = distance_with_numpy(now_frame, ex_frame)
        print("dis", dis)
        if dis > 0.6:
            # print("enouch")
            start = time.clock()

            rgb_frame = now_frame[:, :, ::-1]

            predictions = predict(rgb_frame, model_path=curPath + "/demo_django/sq_face_recignition/model/trained_knn_model.clf")\


            i = 0
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                if name == 'unknown':
                    cv2.imwrite(curPath + '/statics/resource/unknown_face/' + str(uuid.uuid1()) + '.jpg', now_frame[top:bottom, left:right])
                    i += 1
                elif name not in list_knownface[:0]:
                    list_knownface.append((name, int(frame + 0.5), now_frame))
                    # print(list_knownface)



            frames.append(now_frame)
            # print(frame)
            frame += fps
            end = time.clock()
            print("time:{}s".format(end-start))
            continue

        else:
            print("same frame")
            frame += fps
            # end = time.clock()
            # print("time:{}s".format(end - start))
            continue

    print("frames_len", len(frames))

    list_set = set()
    print("list_knownface", list_knownface)
    for name, t, id in list_knownface:
        # print("name", name)
        list_set.add(name)
        # print(list_set)

    # 已知人脸保存时间戳及对应帧
    res_list = []
    localtime = time.time()
    # print("list_set", list_set)
    for item in list_set:
        # print("item", item)
        li_time = []
        li_frame = []
        for list_item in list_knownface:
            if list_item[0] == item:
                li_time.append(list_item[1])
                li_frame.append(list_item[2])
        k = 1
        while k < len(li_time):
            if li_time[k] - li_time[k - 1] <= (fps * 5):
                li_time.remove(li_time[k])
                del li_frame[k]
                k -= 1
            k += 1
        list_name = []
        cnt = 0
        for i in range(len(li_time)):
            seconds = int(li_time[i]/fps+0.5)
            img_name = item + '_' + str(uuid.uuid1()) + '.jpg'
            list_name.append(img_name)

            li_time[i] = seconds
            cv2.imwrite(curPath + '/statics/resource/face_images/' + img_name, li_frame[cnt])
            cnt += 1
        print("(item, li_time, list_name)", (item, li_time, list_name))
        res_list.append((item, li_time, list_name))
    print("res_list", res_list)
    np.save(file, np.array(res_list))
    capture.release()


if __name__ == '__main__':
    file = './test2.mp4'
    main(file)
    pass

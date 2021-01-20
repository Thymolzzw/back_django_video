import os
import cv2
import numpy as np
from skimage.io import imread
from demo_django.face_recignition.facerecoginition_knn import *
import time

def extract_image(path, num):  # 抗裁剪解水印时调用，将原始视频抽帧
    file_name = os.getcwd() + r"/frame_" + str(num) + ".png"
    file_name = './time_allframe/' + str(num) + ".png"
    cap = cv2.VideoCapture(path)  # 读入文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)  # 从num帧开始读视频
    success, frame = cap.read()
    cv2.imwrite(file_name, frame)
    cap.release()
    return file_name


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


def distance(image1, image2):
    image1 = read(image1)
    image2 = read(image2)
    norm_diff = np.linalg.norm(image1 - image2)
    norm1 = np.linalg.norm(image1)
    norm2 = np.linalg.norm(image2)
    return norm_diff / (norm1 + norm2)


def main(video_file_path, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    output_movie = cv2.VideoWriter('testoutput2.mp4', fourcc, 29.97, (1280, 720))
    # classifier = train("./datanew/train", model_save_path="model/trained_knn_model.clf", n_neighbors=2)
    capture = cv2.VideoCapture(video_file_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("frame_count:", frame_count, " fps:", fps)
    frame = 0.00
    frames = []
    list_knownface = []
    while int(frame + 0.5) < frame_count:
        # 先抽出来
        now_frame_path = extract_image(video_file_path, int(frame + 0.5))
        # 再比对特征

        if len(frames) < 1:
            print(frame)
            frame += fps
            frames.append(now_frame_path)
            continue
        ex_frame = frames[-1]
        if distance(now_frame_path, ex_frame) > 0.6:
            start = time.clock()
            print("开始识别{}，当前帧时间：{}s".format(now_frame_path, str(round(frame / fps))))
            # print("在此处调用目标检测代码，传参为当前图片：now_frame_path")

            now_frame = cv2.imread(now_frame_path)
            rgb_frame = now_frame[:, :, ::-1]
            predictions = predict(rgb_frame, model_path="model/trained_knn_model.clf")\
            # STEP 2: Using the trained classifier, make predictions for unknown images
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance

            i = 0
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                if name == 'unknown':
                    # UnKnownfacce=Image.fromarray(UnKnownface)
                    # UnKnownfacce.show()
                    cv2.imwrite('Unknown_face/' + str(i) + '.jpg', now_frame[top:bottom, left:right])
                    i += 1
                elif name not in list_knownface[:0]:
                    list_knownface.append((name, int(frame + 0.5), now_frame))
                    # print(list_knownface)

            if predictions:
                # print("11111")
                # Display results overlaid on an image
                imagefinal = show_prediction_labels_on_image(now_frame, predictions)
                print("Writing frame {} / {}".format(frame, frame_count))
                output_movie.write(np.array(imagefinal))
            else:
                print("Writing frame {} / {}".format(frame, frame_count))
                output_movie.write(frame)

            frames.append(now_frame_path)
            # print(frame)
            frame += fps
            end = time.clock()
            print("time:{}s".format(end-start))
            continue
        else:
            print("same frame")
            os.remove(now_frame_path)
            frame += fps
            # end = time.clock()
            # print("time:{}s".format(end - start))
            continue

    list_set = set()
    for name, t, id in list_knownface:
        list_set.add(name)
        # print(list_set)
    # 统计已知人脸姓名关联百度百科
    # for every in iter(list_set):
    # baidu.query(every)
    # 已知人脸保存时间戳及对应帧
    res_list = []
    for item in list_set:
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
        for i in li_time:
            img_name = 'time_frame/' + item + str(i) + '.jpg'
            list_name.append(item + str(i) + '.jpg')
            img_item = cv2.imwrite(img_name, li_frame[cnt])
            cnt += 1

        res_list.append((item, li_time, list_name))

    print("res_list", res_list)

    np.save(output_path, np.array(res_list))
    datares = np.load(output_path, allow_pickle=True)
    print(datares)
    capture.release()


if __name__ == '__main__':
    file = 'test2.mp4'
    main(file)
    print('done')

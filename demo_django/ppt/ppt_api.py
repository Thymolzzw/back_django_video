import json
import os
import time
import uuid

import face_recognition
import cv2
import numpy as np
from PIL import Image
from skimage.io import imread


def cur_path():
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0]+"demo_django"
    return curPath

def extract_image(path, num):  # 将原始视频抽帧
    name = path.split('/')[-1].split('.')[0]
    file_name = cur_path() + "/statics/resource/ppt_images/" + name + "_" + str(uuid.uuid1()) + "_" + str(int(time.time())) + ".png"
    cap = cv2.VideoCapture(path)  # 读入文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, num)  # 从num帧开始读视频
    success, frame = cap.read()
    # print("frame", frame)
    try:
        cv2.imwrite(file_name, frame)
    except:
        file_name = None
        pass
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

def dHash(img):
    # 差值哈希算法
    # 先将图片压缩成9*8的小图，有72个像素点
    img = cv2.resize(img, (9, 8))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    hash_str = ''

    # 计算差异值：dHash算法工作在相邻像素之间，这样每行9个像素之间产生了8个不同的差异，一共8行，则产生了64个差异值，或者是32位01字符串
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str


def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def distance(image1, image2):
    # image1 = read(image1)
    # image2 = read(image2)
    # norm_diff = np.linalg.norm(image1 - image2)
    # norm1 = np.linalg.norm(image1)
    # norm2 = np.linalg.norm(image2)
    # return norm_diff / (norm1 + norm2)

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    image1_hash = dHash(image1)
    image2_hash = dHash(image2)
    return cmpHash(image1_hash, image2_hash)




def face_detect(filename):
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    face_cascade = cv2.CascadeClassifier(curPath + '/demo_django/ppt/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(curPath + '/demo_django/ppt/haarcascade_eye.xml')
    # 读取图片
    img = cv2.imread(filename)
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(3,3))
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(3, 3))
    # 绘制人脸矩形框
    if len(faces) > 0 or len(eyes) > 0:
        return True
    else:
        return False

    # image = face_recognition.load_image_file(filename)
    # face_locations = face_recognition.face_locations(image)
    # # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    # print("I found {} face(s) in this photograph.".format(len(face_locations)))
    # if len(face_locations) > 0:
    #     return True
    # else:
    #     return False

def distance_is_away(frames, now_frame_path):
    res = True
    for frame in frames:
        if(distance(frame, now_frame_path) < 15):
            res = False
            break
    return res


def main(video_file_path):
    capture = cv2.VideoCapture(video_file_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    # print("frame_count:", frame_count, " fps:", fps)
    frame = 0.00
    frames = []
    while int(frame + 0.5) < frame_count:
        # 先抽出来
        now_frame_path = extract_image(video_file_path, int(frame + 0.5))
        if now_frame_path == None:
            frame += fps
            continue
        # 再比对特征
        if len(frames) < 1:
            # print(frame)
            frame += fps
            frames.append(now_frame_path)
            continue
        ex_frame = frames[-1]

        if distance_is_away(frames, now_frame_path):
            # 检测人脸
            if face_detect(now_frame_path):
                os.remove(now_frame_path)
                frame += fps
                continue
            frames.append(now_frame_path)
            # print(frame)
        else:
            # print("same frame")
            os.remove(now_frame_path)

        frame += fps

    capture.release()

    img_list = []
    list_json = {}
    i = 0
    for img in frames:
        image = Image.open(img)
        img_list.append(image)
        # print(img)
        list_json[str(i)] = img.split("/")[-1]
        i += 1
    # print(img_list)
    # print(list_json)

    json_file_path = os.path.join("statics", "resource", "ppt_json", str(uuid.uuid1())+"_"+str(int(time.time()))+".json")
    with open(os.path.join(cur_path(), json_file_path), "w", encoding='utf-8') as f:
        json.dump(list_json, f)

    pdf_path = None
    if len(img_list) > 0:
        width = 0
        height = 0
        for img in img_list:
            # 单幅图像尺寸
            w, h = img.size
            height += h
            # 取最大的宽度作为拼接图的宽度
            width = max(width, w)
        # 创建空白长图
        result = Image.new(img_list[0].mode, (width, height), 0xffffff)
        # 拼接图片
        height = 0
        for img in img_list:
            w, h = img.size
            # 图片水平居中
            result.paste(img, box=(round(width / 2 - w / 2), height))
            height += h
        # result.show()

        # 保存图片
        pdf_path = os.path.join("statics", "resource", "ppt_pdf",
                                      str(uuid.uuid1()) + "_" + str(int(time.time())) + ".pdf")
        result.save(os.path.join(cur_path(), pdf_path))
    return json_file_path, pdf_path



def ppt_api(file):
    return main(file)


if __name__ == '__main__':
    # main(file)
    # pass
    # cur_path()
    file = "D:\TensorFlow_workplace_new\demo_django\statics\\resource\\videos\\lesson1.mp4"
    ppt_api(file=file)
    # CatchUsbVideo()
    # dis = distance("D:\TensorFlow_workplace_new\demo_django\statics\\resource\ppt_images\\frame_aae2cd74-59af-11eb-b1c8-b48655f33ff9_1610989658.png"
    #          ,"D:\TensorFlow_workplace_new\demo_django\statics\\resource\ppt_images\\frame_c413bf58-59af-11eb-93f9-b48655f33ff9_1610989700.png")
    # print(dis)

    # now = time.time()
    # face_detect("D:\TensorFlow_workplace_new\demo_django\statics\\resource\ppt_images\\frame_aae2cd74-59af-11eb-b1c8-b48655f33ff9_1610989658.png")
    # print(time.time()-now)
    # print(cur_path())
    # print(os.path.join(cur_path(), "statics", "resource", "ppt_json", str(uuid.uuid1())+"_"+str(int(time.time()))+".json"))
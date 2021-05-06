import json

import cv2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import load_img
import numpy as np

class ImgModal:
    base_model = ResNet50(weights='imagenet', include_top=True)
    def __init__(self):
        pass

    def getFeature(self, np_img):
        # 加载图像
        # img = load_img("oneone.jpg", target_size=(224, 224))
        # img = image.img_to_array(img) / 255.0
        img = np.expand_dims(np_img, axis=0)
        predictions = ImgModal.base_model.predict(img)
        return predictions[0]

    def doImgModal(self, video_path='ttt.mp4', result_path='result.json', frame_inter=10):
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        # print("frame_count:", frame_count, " fps:", fps)
        frame = 0.00
        results = {}
        while int(frame + 0.5) < frame_count:
            # 先抽出来
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame + 0.5))  # 从num帧开始读视频
            success, now_frame_path = capture.read()
            if not success:
                frame += fps * frame_inter
                continue
            else:
                # img = load_img("oneone.jpg", target_size=(224, 224))
                # img = image.img_to_array(img) / 255.0
                img = cv2.resize(now_frame_path, (224, 224), interpolation=cv2.INTER_CUBIC) / 255.0
                f = int(frame + 0.5)
                fe = self.getFeature(np_img=img)
                results[str(int(f/fps))] = fe.tolist()
                print(results)
            frame += fps * frame_inter
        capture.release()
        with open(result_path, "w", encoding='utf-8') as f:
            json.dump(results, f)

    def searchFeature(self, result_path, search_feature=None, yuzhi=0.01):
        with open(result_path, 'r') as load_f:
            data = json.load(load_f)
        result = []
        for item in data:
            pass
            # print(item)
            dist = np.sqrt(np.sum(np.square(data[item] - search_feature)))
            # print(dist)
            if dist < yuzhi:
                temp = {}
                temp[item] = dist
                result.append(temp)
        return result

# np.save("oneone.npy", predictions[0])



# This is a sample Python script.
import json
import os

import cv2
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import load_img

from demo_django.CrossModal.ImgModal import ImgModal



if __name__ == '__main__':
    pass
    imgModal = ImgModal()

    img_path = 'wed4.jpg'
    img = load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    f = imgModal.getFeature(img)
    res = imgModal.searchFeature('result_wed.json', f, yuzhi=0.015)
    print(res)

    # imgModal.doImgModal(video_path='wed.mp4', result_path='result_wed.json', frame_inter=5)



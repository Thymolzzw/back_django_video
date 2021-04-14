import json
import mimetypes
import os
import sys
import time
import uuid
from datetime import datetime
import json

import cv2
import pytz
from PIL import Image
from django.core import serializers
import numpy as np
from django.http import HttpResponse, JsonResponse, response
from django.shortcuts import render, redirect
from MyModel.models import Users, Videos, Binner, People, SourceInformation, Country
import hashlib

# from demo_django.AdelaiDet.TextRecognize_API import text_recognize
from demo_django.asr.pyTranscriber.asr_api import asr_subtitle

# from demo_django.darknet.ObjectDetection_API import objectDetection
from demo_django.product.OCR_report import ocr_report_return_with_path
from demo_django.utils import getBaseUrl, md5Encode

import re
from wsgiref.util import FileWrapper
from django.http import StreamingHttpResponse

# zzw
from demo_django.wiki.wiki import wiki_api

import base64
from io import BytesIO

# 视频流
def file_iterator(file_name, chunk_size=8192, offset=0, length=None):
    with open(file_name, "rb") as f:
        f.seek(offset, os.SEEK_SET)
        remaining = length
        while True:
            bytes_length = chunk_size if remaining is None else min(remaining, chunk_size)
            data = f.read(bytes_length)
            if not data:
                break
            if remaining:
                remaining -= len(data)
            yield data


def stream_video(request, path):
    """将视频文件以流媒体的方式响应"""
    range_header = request.META.get('HTTP_RANGE', '').strip()
    range_re = re.compile(r'bytes\s*=\s*(\d+)\s*-\s*(\d*)', re.I)
    range_match = range_re.match(range_header)
    size = os.path.getsize(path)
    content_type, encoding = mimetypes.guess_type(path)
    # print("content_type", content_type)
    # print("encoding", encoding)
    content_type = content_type or 'application/octet-stream'
    if range_match:
        first_byte, last_byte = range_match.groups()
        first_byte = int(first_byte) if first_byte else 0
        last_byte = first_byte + 1024 * 1024 * 8  # 8M 每片,响应体最大体积
        if last_byte >= size:
            last_byte = size - 1
        length = last_byte - first_byte + 1
        # print("first_byte", first_byte)
        # print("length", length)
        resp = StreamingHttpResponse(file_iterator(path, offset=first_byte, length=length), status=206,
                                     content_type=content_type)
        resp['Content-Length'] = str(length)
        resp['Content-Range'] = 'bytes %s-%s/%s' % (first_byte, last_byte, size)
    else:
        # 不是以视频流方式的获取时，以生成器方式返回整个文件，节省内存
        resp = StreamingHttpResponse(FileWrapper(open(path, 'rb')), content_type=content_type)
        resp['Content-Length'] = str(size)
    resp['Accept-Ranges'] = 'bytes'
    # print("resp", resp)
    return resp

def streamVideo(request):
    videoId = request.GET.get("videoId")
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    path = os.path.join(curPath, video.rel_path)
    # if video.border_video_path != None and video.border_video_path != "":
    #     path = os.path.join(curPath, video.border_video_path)

    # print("path", path)
    # path = '/home/wh/zzw/demo_django/statics/resource/videos/7c1deada-5e41-11eb-b8f4-b0a460e7b2fb_7f8343dc-5e41-11eb-b8f4-b0a460e7b2fb_result.mp4'
    # print("path", path)
    return stream_video(request, path)

def getAllVideos(request):
    video_lists = Videos.objects.filter(is_delete=0)
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    for video in video_lists:
        # 没有缩略图会自动生成
        if video.snapshoot_img != None and video.snapshoot_img != "" and os.path.exists(curPath + '/' + video.snapshoot_img):
            video.snapshoot_img = getBaseUrl() + video.snapshoot_img
        else:
            # 抽帧作为缩略图
            image_path = os.path.join(curPath, 'statics', 'resource', 'images', video.name.split(".")[0] + ".png")
            image_path_db = 'statics/' + 'resource/' + 'images/' + video.name.split(".")[0] + ".png"
            video_path = os.path.abspath(os.path.join(curPath, 'statics', 'resource', 'videos', video.name))
            extract_image(video_path, image_path=image_path)
            video.snapshoot_img = image_path_db
            video.save()
            video.snapshoot_img = getBaseUrl() + video.snapshoot_img

        # create time formate
        video.rel_path = getBaseUrl() + video.rel_path

        if video.text_location != None:
            video.text_location = getBaseUrl() + video.text_location
        if video.subtitle != None:
            video.subtitle = getBaseUrl() + video.subtitle
        if video.asr_path != None:
            video.asr_path = getBaseUrl() + video.asr_path
        video.create_time = formate_time(video.create_time)
    videos = serializers.serialize("json", video_lists)


    return HttpResponse(JsonResponse({
        "code": 0,
        "msg": "",
        "status": 1,
        "data": json.loads(videos)
    }), content_type="application/json")

def formate_time(t):
    timeArray = time.localtime(t)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime

def parseVTT(filepath):
    subTitle = []
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
        if "WEBVTT" in line:
            line = f.readline()
            line = f.readline()
        while line:
            item = {}
            line = f.readline()
            item["time"] = line
            line = f.readline()
            while True:
                l = f.readline()
                if l != '\n':
                    line = line + l
                else:
                    break
            item["content"] = line
            line = f.readline()
            while line == '\n':
                line = f.readline()
            subTitle.append(item)
    # print(subTitle)
    return subTitle

def getSubTitle(request):
    videoId = request.GET.get("videoId")
    # print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 字幕
    subTitle = []
    if video.translate_subtitle != None and video.translate_subtitle != ""\
            and os.path.exists(os.path.join(curPath, video.translate_subtitle)):
        subTitle = parseVTT(os.path.join(curPath, video.translate_subtitle))

    print('subTitle', subTitle)
    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["subTitle"] = subTitle
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def parseVTTForUpdate(filepath):
    subTitle = []
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.readline()
        if "WEBVTT" in line:
            line = f.readline()
            line = f.readline()
        while line:
            item = {}
            line = f.readline()
            item["time"] = line.replace('\n', '')
            line_item  = []
            line = f.readline()
            while True:
                l = f.readline()
                if l != '\n':
                    line = line + l
                else:
                    break
            item["content"] = line[:-1].split('\n')
            line = f.readline()
            while line == '\n':
                line = f.readline()
            subTitle.append(item)
    # print(subTitle)
    return subTitle



def getSubTitleForUpdate(request):
    videoId = request.GET.get("videoId")
    # print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 字幕
    subTitle = []
    if video.translate_subtitle != None and video.translate_subtitle != ""\
            and os.path.exists(os.path.join(curPath, video.translate_subtitle)):
        subTitle = parseVTTForUpdate(os.path.join(curPath, video.translate_subtitle))

    # print('subTitle', subTitle)
    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["subTitle"] = subTitle
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def sizeConvert(size):
    K, M, G = 1024, 1024 ** 2, 1024 ** 3
    if size >= G:
        return str(round(size/G,2)) + 'G Bytes'
    elif size >= M:
        return str(round(size/M,2)) + 'M Bytes'
    elif size >= K:
        return str(round(size/K,2)) + 'K Bytes'
    else:
        return str(size) + 'Bytes'


def hcf(x, y):
    """该函数返回两个数的最大公约数"""
    x = int(x)
    y = int(y)
    if x > y:
        smaller = y
    else:
        smaller = x
    hcf = 1
    for i in range(1, smaller + 1):
        if ((x % i == 0) and (y % i == 0)):
            hcf = i

    return hcf

def getVideoAdditionData(request):

    videoId = request.GET.get("videoId")
    # print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 附加信息
    addition_data = {}

    addition_data["video_introduce"] = video.introduce
    # print(video.introduce)

    addition_data["video_title"] = video.title

    video_time = get_video_times(os.path.join(curPath, video.rel_path))
    addition_data["video_time"] = video_time

    # file size
    file_byte = os.path.getsize(os.path.join(curPath, video.rel_path))
    addition_data["video_file_size"] = sizeConvert(file_byte)

    # file fps
    v = cv2.VideoCapture(os.path.join(curPath, video.rel_path))
    fps = v.get(cv2.CAP_PROP_FPS)
    addition_data["video_fps"] = fps

    # file width height
    width = v.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = v.get(cv2.CAP_PROP_FRAME_HEIGHT)
    addition_data["video_frame_width"] = int(width)
    addition_data["video_frame_height"] = int(height)

    # file width height proportion
    the_hcf = hcf(width, height)
    # addition_data["the_hcf"] = the_hcf
    width = int(width) / the_hcf
    height = int(height) / the_hcf
    addition_data["video_frame_proportion"] = str(int(width)) + '×' + str(int(height))


    if video.create_user != None:
        user = Users.objects.filter(id=video.create_user).first()
        if user:
            addition_data["create_user"] = user.name
        else:
            addition_data["create_user"] = None

    if video.source_id != None:
        source = SourceInformation.objects.filter(id=video.source_id).first()
        if source:
            addition_data["source_name"] = source.name
            addition_data["source_intro"] = source.introduce
            addition_data["source_url"] = source.source_url
        else:
            addition_data["source_name"] = None
            addition_data["source_intro"] = None
            addition_data["source_url"] = None

    if video.country_id != None:
        country = Country.objects.filter(id=video.country_id).first()
        if country:
            addition_data["country_name"] = country.name
        else:
            addition_data["country_name"] = None

    functions = []
    if video.face_npy_path != None and video.face_npy_path != "":
        functions.append(0)
    if video.equipment_json_path != None and video.equipment_json_path != "":
        functions.append(1)
    if video.ppt_pdf_path != None and video.ppt_pdf_path != "":
        functions.append(2)
    if video.text_location != None and video.text_location != "":
        functions.append(3)
    if video.translate_subtitle != None and video.translate_subtitle != "":
        functions.append(4)
    if video.voice_json != None and video.voice_json != "":
        functions.append(5)

    tag_list = None
    if video.tag != None and video.tag != "":
        tag_list = video.tag.strip().split(' ')
    print(tag_list)
    addition_data["video_tag"] = tag_list

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["addition_data"] = addition_data
    resp["functions"] = functions
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getCountryList(request):
    # pass
    countryList = Country.objects.all()

    # countryList = serializers.serialize("json", countryList)
    countrys = []
    for country in countryList:
        temp = {}
        temp['country_id'] = country.id
        temp['country_name'] = country.name
        temp['country_introduce'] = country.introduce
        countrys.append(temp)

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = countrys
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getResourceList(request):
    # pass
    sourceList = SourceInformation.objects.filter(is_delete=0)

    # countryList = serializers.serialize("json", countryList)
    resources = []
    for resource in sourceList:
        temp = {}
        temp['resource_id'] = resource.id
        temp['resource_name'] = resource.name
        temp['resource_introduce'] = resource.introduce
        temp['resource_url'] = resource.source_url
        resources.append(temp)

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = resources
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def deleteResource(request):
    delete_resource_id = request.POST.get('delete_resource_id')
    resource = SourceInformation.objects.filter(id= delete_resource_id).first()
    code = 20000
    try:
        resource.is_delete = 1
        resource.save()
    except:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def deleteCountry(request):
    delete_country_id = request.POST.get('delete_country_id')
    print('delete_country_id', delete_country_id)
    country = Country.objects.filter(id= delete_country_id).first()
    code = 20000
    try:
        country.is_delete = 1
        country.save()
    except:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def addEditResource(request):
    code = 20000
    resource = None
    try:
        dialog_status = request.POST.get('dialog_status')
        if dialog_status == 'add':
            #添加数据源
            resource_name = request.POST.get('resource_name')
            resource_introduce = request.POST.get('resource_introduce')
            resource_url = request.POST.get('resource_url')
            resourceList = SourceInformation.objects.filter(name=resource_name, is_delete=0)
            if len(resourceList) > 0:
                code = 3000
            else:
                resource = SourceInformation.objects.create(name=resource_name, introduce=resource_introduce, source_url=resource_url, is_delete=0)
        elif dialog_status == 'edit':
            #修改数据源
            resource_id = request.POST.get('resource_id')
            resource_name = request.POST.get('resource_name')
            resource_introduce = request.POST.get('resource_introduce')
            resource_url = request.POST.get('resource_url')
            resource = SourceInformation.objects.filter(id=resource_id).first()
            resource.name = resource_name
            resource.introduce = resource_introduce
            resource.source_url = resource_url
            resource.save()
        else:
            code = 2000

        if resource:
            # resource不为None
            temp = {}
            temp['resource_id'] = resource.id
            temp['resource_name'] = resource.name
            temp['resource_introduce'] = resource.introduce
            temp['resource_url'] = resource.source_url
            resource = temp

    except:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = resource
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def addEditCountry(request):
    code = 20000
    country = None
    try:
        dialog_status = request.POST.get('dialog_status')
        if dialog_status == 'add':
            #添加数据源
            country_name = request.POST.get('country_name')
            country_introduce = request.POST.get('country_introduce')
            countryList = Country.objects.filter(name=country_name, is_delete=0)
            if len(countryList) > 0:
                code = 3000
            else:
                country = Country.objects.create(name=country_name, introduce=country_introduce, is_delete=0)
        elif dialog_status == 'edit':
            #修改数据源
            country_id = request.POST.get('country_id')
            country_name = request.POST.get('country_name')
            country_introduce = request.POST.get('country_introduce')
            country = Country.objects.filter(id=country_id).first()
            country.name = country_name
            country.introduce = country_introduce
            country.save()
        else:
            code = 2000

        if country:
            # resource不为None
            temp = {}
            temp['country_id'] = country.id
            temp['country_name'] = country.name
            temp['country_introduce'] = country.introduce
            country = temp

    except:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = country
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getVideoEquipment(request):
    videoId = request.GET.get("videoId")
    # print(videoId)
    video = Videos.objects.filter(id=videoId).first()
    name_dict = {}
    name_dict['helicopter'] = '直升机'
    name_dict['military ship'] = '水面舰艇'
    name_dict['armored car'] = '装甲车'
    name_dict['fighter'] = '战斗机'

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    equipment_json_data = None
    if video.equipment_json_path != None and video.equipment_json_path != ""\
            and os.path.exists(os.path.join(curPath, video.equipment_json_path)):
        with open(os.path.join(curPath, video.equipment_json_path), 'r') as load_f:
            load_dict = json.load(load_f)

            # print(load_dict)
            for item in load_dict:
                # print(item)
                base_image_path = os.path.join(curPath, "statics", "resource", "object_images", item["filename"])
                item["filename"] = getBaseUrl() + "statics/resource/object_images/" + item["filename"]

                image_old = cv2.imread(base_image_path)
                image_old_width = image_old.shape[:2][1]
                image_old_height = image_old.shape[:2][0]

                for item_item in item["objects"]:
                    # print(item_item["relative_coordinates"])
                    center_x = item_item["relative_coordinates"]["center_x"]
                    center_y = item_item["relative_coordinates"]["center_y"]

                    width = item_item["relative_coordinates"]["width"]
                    height = item_item["relative_coordinates"]["height"]

                    if width > 1.0:
                        width = 0.99
                    if height > 1.0:
                        height = 0.99


                    # # 异常图片
                    # if width > 1.0 or height > 1.0:
                    #     item_item["image"] = None
                    #     continue

                    width = width*image_old_width
                    height = height*image_old_height

                    # print("center_x", center_x)
                    # print("center_y", center_y)

                    center_x_ps = image_old_width*center_x
                    center_y_ps = image_old_height*center_y
                    # print("center_x_ps", center_x_ps)
                    # print("center_y_ps", center_y_ps)
                    # print("width", width)
                    # print("height", height)

                    coor = []
                    coor.append(center_y_ps - height / 2.0)
                    coor.append(center_x_ps - width/2.0)
                    coor.append(center_y_ps + height / 2.0)
                    coor.append(center_x_ps + width / 2.0)

                    if coor[2] > image_old_height:
                        coor[2] = image_old_height

                    if coor[3] > image_old_width:
                        coor[3] = image_old_width

                    # print("coor", coor)

                    # print("image_old", image_old.shape[:2])
                    image_cut = image_old[max(int(coor[0])+1, 0):max(int(coor[2])+1, 0), max(int(coor[1]), 0):max(int(coor[3]), 0), :]
                    # print("image_cut", image_cut)
                    try:
                        item_item['name'] = name_dict[item_item['name']]
                    except:
                        pass
                    item_item['image'] = 'data:image/png;base64,' + image_to_base64(image_cut)

                # print(item["objects"])

            equipment_json_data = load_dict

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    resp["equipment_json_data"] = equipment_json_data
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def image_to_base64(image_np):
    # print("image_np", image_np)
    retval, buffer = cv2.imencode('.jpg', image_np)
    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()
    return pic_str

def getVideoPPT(request):
    videoId = request.GET.get("videoId")
    # print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    ppt_json = None
    if video.ppt_json_path != None and video.ppt_json_path != "":
        with open(os.path.join(curPath, video.ppt_json_path), 'r') as load_f:
            load_dict = json.load(load_f)
            # print(load_dict)

            for item in load_dict:
                # print(item)
                load_dict[item] = getBaseUrl() + "statics/resource/ppt_images/" + load_dict[item]
            ppt_json = load_dict

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["ppt_json"] = ppt_json
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def deleteFaceItem(request):
    videoId = request.POST.get("videoId")
    update_people_index = request.POST.get("delete_people_index")
    update_time_index = request.POST.get("delete_time_index")
    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    if video.face_npy_path != None and video.face_npy_path != "" \
            and os.path.exists(os.path.join(curPath, video.face_npy_path)):
        loadData = np.load(os.path.join(curPath, video.face_npy_path)
                           , allow_pickle=True)
        loadData = loadData.tolist()

        # 删除
        if len(loadData[int(update_people_index)][1]) == 1:
            # 长度为1，删除整体
            del loadData[int(update_people_index)]
            # loadData = np.delete(loadData, int(update_people_index), axis=0)
            pass
        else:
            # 长度大于1，删除对应对象
            del loadData[int(update_people_index)][1][int(update_time_index)]
            del loadData[int(update_people_index)][2][int(update_time_index)]
            pass

        # print("new", loadData)
        # print(video.face_npy_path)
        np.save(os.path.join(curPath, video.face_npy_path), np.array(loadData))

    pass
    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def updateFaceItem(request):
    videoId = request.POST.get("videoId")
    update_people_index = request.POST.get("update_people_index")
    update_time_index = request.POST.get("update_time_index")
    update_time_people_name = request.POST.get("update_time_people_name")
    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    if video.face_npy_path != None and video.face_npy_path != "" \
            and os.path.exists(os.path.join(curPath, video.face_npy_path)):
        loadData = np.load(os.path.join(curPath, video.face_npy_path)
                           , allow_pickle=True)
        loadData = loadData.tolist()
        # print("old", loadData)
        find_people_index = -1
        for i in range(len(loadData)):
            if loadData[i][0] == update_time_people_name:
                find_people_index = i
                break
        if find_people_index == -1:
            # 文件中不存在该人名
            pass
            new_item = []
            new_item.append(update_time_people_name)
            new_item_item = []
            new_item_item.append(loadData[int(update_people_index)][1][int(update_time_index)])
            new_item.append(new_item_item)
            new_item_item_item = []
            new_item_item_item.append(loadData[int(update_people_index)][2][int(update_time_index)])
            new_item.append(new_item_item_item)

            loadData.append(new_item)
            # loadData = np.array(loadData)
            # loadData = np.append(loadData, np.array((update_time_people_name, new_item_item, new_item_item_item)))
            # loadData.insert(len(loadData), new_item)
            # print("old2", loadData.tolist())
        else:
            # 存在该人名
            # 插入
            insert_index = 0
            t = loadData[int(update_people_index)][1][int(update_time_index)]
            for i in range(len(loadData[find_people_index][1])):
                # print("ok", i)
                if int(loadData[find_people_index][1][i]) < int(t):
                    insert_index = i
                    pass
                else:
                    insert_index = i
                    break
            # print(insert_index)
            if loadData[find_people_index][1][insert_index] != t:
                # 插入
                if insert_index == len(loadData[find_people_index][1]) - 1:
                    loadData[find_people_index][1].insert(insert_index + 1,
                                                          loadData[int(update_people_index)][1][int(update_time_index)])
                    loadData[find_people_index][2].insert(insert_index + 1,
                                                          loadData[int(update_people_index)][2][int(update_time_index)])
                else:
                    loadData[find_people_index][1].insert(insert_index,
                                                          loadData[int(update_people_index)][1][int(update_time_index)])
                    loadData[find_people_index][2].insert(insert_index,
                                                          loadData[int(update_people_index)][2][int(update_time_index)])

        # 删除
        if len(loadData[int(update_people_index)][1]) == 1:
            # 长度为1，删除整体
            del loadData[int(update_people_index)]
            # loadData = np.delete(loadData, int(update_people_index), axis=0)
            pass
        else:
            # 长度大于1，删除对应对象
            del loadData[int(update_people_index)][1][int(update_time_index)]
            del loadData[int(update_people_index)][2][int(update_time_index)]
            pass

        # print("new", loadData)
        print(video.face_npy_path)
        np.save(os.path.join(curPath, video.face_npy_path), np.array(loadData))

    pass
    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getFace(request):
    videoId = request.GET.get("videoId")
    # print("getface")
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 人脸识别
    face_data = None
    if video.face_npy_path != None and video.face_npy_path != ""\
            and os.path.exists(os.path.join(curPath, video.face_npy_path)):
        print(video.face_npy_path)
        loadData = np.load(os.path.join(curPath, video.face_npy_path)
                           , allow_pickle=True)

        # print(loadData)
        face_data = []
        for item in loadData:
            # print(item)
            item_item = {"name": item[0]}
            item_list = []
            for i in range(len(item[1])):
                item[2][i] = getBaseUrl() + "statics/" + "resource/" + "face_images/" + item[2][i]
                item_list.append({
                    'time': item[1][i],
                    'img': item[2][i]
                })
            # print(item_list)
            item_item["time_img"] = item_list

            # 人物介绍
            # print("name:", item_item["name"])
            people = People.objects.filter(name=item_item["name"]).first()
            if people != None:
                if people.introduce == None or people.introduce == "":
                    introduce = None
                    try:
                        # introduce = wiki_api(people.name)
                        pass
                    except:
                        pass
                    item_item["introduce"] = introduce
                    if introduce != None and introduce != "":
                        people.introduce = introduce
                        people.save()
                else:
                    item_item["introduce"] = people.introduce

                if people.img != None and people.img != "":
                    item_item["head_img"] = getBaseUrl() + people.img
                else:
                    item_item["head_img"] = getBaseUrl() + "statics/resource/images/head_default.jpg"
            else:
                item_item["head_img"] = getBaseUrl() + "statics/resource/images/head_default.jpg"
                item_item["introduce"] = None
                people = People.objects.create(name=item_item["name"], is_delete=0)
                introduce = None
                try:
                    # introduce = wiki_api(people.name)
                    pass
                except:
                    pass
                item_item["introduce"] = introduce
                if introduce != None and introduce != "":
                    people.introduce = introduce
                    people.save()
                pass

            face_data.append(item_item)


    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["face_data"] = face_data
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getText(request):
    videoId = request.GET.get("videoId")
    # print("进入----------------------------------------------")
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 文本识别
    text_data = None
    if video.text_location != None and video.text_location != "":
        try:
            with open(os.path.join(curPath, video.text_location), 'r') as load_f:
                text_data = json.load(load_f)
            # print(text_data)
        except:
            pass

    if text_data != None:
        for item in text_data:
            try:
                item["image"] = getBaseUrl() + "statics/resource/text_images/" + item["image"]
            except:
                pass

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["text_data"] = text_data
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getOneVideos(request):

    videoId = request.GET.get("videoId")
    # print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    video.rel_path = getBaseUrl() + video.rel_path

    if video.subtitle != None and video.subtitle != "":
        video.subtitle = getBaseUrl() + video.subtitle
    if video.asr_path != None and video.asr_path != "":
        video.asr_path = getBaseUrl() + video.asr_path
    if video.ppt_pdf_path != None and video.ppt_pdf_path != "":
        video.ppt_pdf_path = getBaseUrl() + video.ppt_pdf_path
    if video.translate_subtitle != None and video.translate_subtitle != "":
        subtitile = str(video.translate_subtitle)
        subtitile = subtitile.replace('statics', 'static')
        video.translate_subtitle = getBaseUrl() + subtitile

    videos = serializers.serialize("json", [video])

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = json.loads(videos)
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def get_video_times(video_path):
    """
    pip install moviepy
    获取指定的视频时长
    """
    from moviepy.editor import VideoFileClip
    video_clip = VideoFileClip(video_path)
    durantion = video_clip.duration
    video_clip.reader.close()
    video_clip.audio.reader.close_proc()
    return durantion


def getAllPeopleFace(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    loadData = np.load(os.path.join(curPath, 'statics', 'resource', 'text', 'res_face.npy')
                       , allow_pickle=True)
    data = []
    for item in loadData:
        # print(item)
        item_item = {"name": item[0]}
        item_list = []
        for i in range(len(item[1])):
            item[2][i] = getBaseUrl()+"statics/"+"resource/"+"images/"+item[2][i]
            item_list.append({
                'time': item[1][i],
                'img': item[2][i]
            })
        # print(item_list)
        item_item["time_img"] = item_list
        data.append(item_item)
    res = {
        "code": 0,
        "msg": "",
        "status": 1,
        "data": data
    }
    # print(res)
    return HttpResponse(JsonResponse(res), content_type="application/json")


def getBinner(request):
    binner_lists = Binner.objects.all()
    for item in binner_lists:
        item.img = getBaseUrl() + item.img
    binners = serializers.serialize("json", binner_lists)
    return HttpResponse(JsonResponse({
        "code": 0,
        "msg": "",
        "status": 1,
        "data": json.loads(binners)
    }), content_type="application/json")


def doLogin(request):
    account_name = request.POST.get("account_name")
    password = request.POST.get("password")
    print("password", password)
    password = md5Encode(password)
    # print(account_name+password)
    user = Users.objects.filter(account_name=account_name, password=password)
    # print(user)
    if len(user) != 0:
        user = serializers.serialize("json", user)
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "成功查找到该用户，该用户存在！",
            "status": 1,
            "data": json.loads(user)
        }), content_type="application/json")
    else:
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "该用户不存在！",
            "status": 1,
            "data": {
                "account_name": account_name,
                "password": password,
            }
        }), content_type="application/json")

def doSignUp(request):
    name = request.GET.get("name")
    account_name = request.GET.get("account_name")
    password = request.GET.get("password")
    password = md5Encode(password)
    # print(password)

    user = Users.objects.filter(name=name)
    if len(user) != 0:
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "用户已经存在！",
            "status": 1,
            "data": {
                "name": name,
                "account_name": account_name,
                "password": password
            }
        }), content_type="application/json")
    else:
        user_save = Users(name=name, account_name=account_name, password=password, type=1)
        user_save.save()
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "用户注册成功！",
            "status": 1,
            "data": {
                "name": name,
                "account_name": account_name
            }
        }), content_type="application/json")


def getUserInfo(request):
    print("jihru")
    account_name = request.POST.get("account_name")
    print("account_name", account_name)
    user = Users.objects.filter(account_name=account_name)
    if len(user) != 0:
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "用户存在！",
            "status": 1,
            "data": {
                "role": user.type
            }
        }), content_type="application/json")
    else:
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "error ！",
            "status": 1,
            "data": {
            }
        }), content_type="application/json")



def extract_image(video_path, image_path):
    cap = cv2.VideoCapture(video_path)  # 读入文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)  # 从num帧开始读视频
    success, frame = cap.read()
    cv2.imwrite(image_path, frame)
    cap.release()

def getAllResource(request):
    source_list = SourceInformation.objects.all()
    source_list = serializers.serialize("json", source_list)
    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = json.loads(source_list)
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getProduct(request):
    videoId = request.POST.get("videoId")
    functions = request.POST.get("functions")

    video = Videos.objects.filter(id=videoId).first()
    product_result = None
    print("videoId", videoId)
    print("functions", functions)
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    if '6' in functions:
        # ocr product
        if video.text_pdf_location != None and video.text_pdf_location != ""\
                and os.path.exists(os.path.join(curPath, video.text_pdf_location)):
            os.remove(os.path.join(curPath,video.text_pdf_location))
            # print("删除test_pdf文件")
            pass
        else:
            # print("test_pdf文件不存在")
            pass
        ocr_pdf_path = ocr_report_return_with_path(os.path.join(curPath, video.text_location))
        print("ocr_pdf_path", ocr_pdf_path)
        if ocr_pdf_path != None:
            video.text_pdf_location = ocr_pdf_path
            # print("ocr_pdf_path", ocr_pdf_path)
            video.save()
        if video.text_pdf_location != None and video.text_pdf_location != "" \
                and os.path.exists(os.path.join(curPath, video.text_pdf_location)):
            product_result = getBaseUrl() + video.text_pdf_location

    if '7' in functions:
        # PPT product
        if video.ppt_pdf_path != None and video.ppt_pdf_path != "" \
                and os.path.exists(os.path.join(curPath,video.ppt_pdf_path)):
            os.remove(os.path.join(curPath, video.ppt_pdf_path))
            # print("删除ppt_pdf文件")
            # pass
            frames = []
            if video.ppt_json_path != None and video.ppt_json_path != "" \
                    and os.path.exists(os.path.join(curPath, video.ppt_json_path)):
                with open(os.path.join(curPath, video.ppt_json_path), 'r') as load_f:
                    load_dict = json.load(load_f)
                    # print(load_dict)

                    for item in load_dict:
                        # print(item)
                        frames.append(os.path.join(curPath, "statics", "resource", "ppt_images", load_dict[item]))
                        # load_dict[item] = getBaseUrl() + "statics/resource/ppt_images/" + load_dict[item]
                    # ppt_json = load_dict
            img_list = []
            i = 0
            for img in frames:
                image = Image.open(img)
                img_list.append(image)
                # print(img)
                i += 1

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
                pdf_path = "statics/" + "resource/" + "ppt_pdf/" + str(uuid.uuid1()) + "_" + str(
                    int(time.time())) + ".pdf"
                result.save(os.path.join(curPath, pdf_path))
            video.ppt_pdf_path = pdf_path
            video.save()
            product_result = getBaseUrl() + video.ppt_pdf_path

    if '8' in functions:
            # card product
            product_result = getBaseUrl() + video.snapshoot_img
            pass


    product = {}
    product['product_result'] = product_result

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = product
    return HttpResponse(JsonResponse(resp), content_type="application/json")



def getVideoVoicePrint(request):
    videoId = request.GET.get("videoId")
    print('videoId', videoId)

    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    # print(curPath)

    load_dict = None
    if video.voice_json != None and video.voice_json != ''\
            and os.path.exists(os.path.join(curPath, video.voice_json)):
        with open(curPath + '/' + str(video.voice_json), 'r', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = load_dict
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def changeVoicePrint(request):
    videoId = request.POST.get("videoId")
    print('videoId', videoId)
    update_json = json.loads(request.POST.get("update_json"))

    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    # print(curPath)


    if video.voice_json != None and video.voice_json != ''\
            and os.path.exists(os.path.join(curPath, video.voice_json)):
        with open(curPath + '/' + str(video.voice_json), 'w', encoding='utf-8') as load_f:
            json.dump(update_json, load_f)

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getPeopleList(request):
    # pass
    sourceList = People.objects.filter(is_delete=0)

    resources = []
    for resource in sourceList:
        temp = {}
        temp['people_id'] = resource.id
        temp['people_img'] = resource.img
        temp['people_introduce'] = resource.introduce
        temp['people_voice_feature_path'] = resource.voice_feature_path
        temp['people_name'] = resource.name
        temp['people_is_delete'] = resource.is_delete
        resources.append(temp)

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = resources
    return HttpResponse(JsonResponse(resp), content_type="application/json")


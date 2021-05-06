import json
import mimetypes
import os
import random
import sys
import time
import uuid
from datetime import datetime
import json
from shutil import copyfile

import cv2
import pytz
from PIL import Image
from django.core import serializers
import numpy as np
from django.http import HttpResponse, JsonResponse, response
from django.shortcuts import render, redirect
from MyModel.models import Users, Videos, Binner, People, SourceInformation, Country, PeopleRelation, Collection, \
    Operations
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

def getHotVideos(request):
    video_lists = Videos.objects.filter(is_delete=0).order_by("view_volume")
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
    resp["code"] = 20000
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
    resp["code"] = 20000
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
    # print(tag_list)
    addition_data["video_tag"] = tag_list

    resp = {}
    resp["code"] = 20000
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
    resp["code"] = 20000
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
    resp["code"] = 20000
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
            resourceList = SourceInformation.objects.filter(name=resource_name)
            if len(resourceList) > 0:
                # 数据库中已经存在
                re = resourceList.first()
                if re.is_delete != 0:
                    # 数据库中该记录被删除， 恢复更新内容
                    re.is_delete = 0
                    re.introduce = resource_introduce
                    re.source_url = resource_url
                    re.save()

            else:
                 # 数据库中不存在
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
    resp["code"] = 20000
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
    reloadPeopleImg()
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
        # print(video.face_npy_path)
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
                        introduce = wiki_api(people.name)
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
                    introduce = wiki_api(people.name)
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
    resp["code"] = 20000
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
    resp["code"] = 20000
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
    account_name = request.POST.get("username")
    password = request.POST.get("password")
    password = md5Encode(password)
    print(account_name, password)
    # print(account_name + "##" + password)
    user = Users.objects.filter(account_name=account_name, password=password, is_delete=0)
    # print(len(user))
    if len(user) != 0:
        user = user.first()

        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "成功查找到该用户，该用户存在！",
            "status": 1,
            "token": str(user.id),
        }), content_type="application/json")
    else:
        return HttpResponse(JsonResponse({
            "code": 2000,
            "msg": "该用户不存在！",
            "status": 1,
            "token": None,
        }), content_type="application/json")

def getUserInfo(request):
    token = request.GET.get("token")
    print("token", token)
    user = Users.objects.filter(id=token)
    if len(user) > 0:
        # 存在该用户
        user = user.first()
        roles = []
        if user.type == 1:
            # 普通用户
            roles.append("visitor")
        elif user.type == 2:
            # 专家
            roles.append("editor")
        elif user.type == 3:
            # 管理员
            roles.append("admin")
        current_user = {
            "id": user.id,
            "name": user.name,
            "account_name": user.account_name,
            "password": user.password,
            "type": user.type,
            "roles": roles,
            "introduce": user.introduce,
            "email": user.email,
            "avatar": get_gravatar_url(user.account_name),
        }
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "该用户不存在！",
            "status": 1,
            "data": current_user,
        }), content_type="application/json")
    else:
        # 不存在该用户
        return HttpResponse(JsonResponse({
            "code": 2000,
            "msg": "该用户不存在！",
            "status": 1,
            "data": None,
        }), content_type="application/json")

def editUserInfo(request):
    user_id = request.POST.get("user_id")
    user_account_name = request.POST.get("user_account_name")
    user_password = request.POST.get("user_password")
    user_real_name = request.POST.get("user_real_name")
    user_introduce = request.POST.get("user_introduce")
    user_email = request.POST.get("user_email")
    # print("user_id", user_id)
    code = 20000
    try:
        user = Users.objects.filter(account_name=user_account_name)
        if len(user) > 0:
            user = user.first()
            if user.id != int(user_id):
                # 账户名已存在
                code = 3000
                return HttpResponse(JsonResponse({
                    "code": code,
                    "msg": "",
                    "status": 1,
                }), content_type="application/json")

        user = Users.objects.filter(id=user_id).first()
        user.account_name = user_account_name
        if user_password != '':
            user.password = md5Encode(user_password)
        user.name = user_real_name
        user.introduce = user_introduce
        user.email = user_email
        user.save()
    except:
        code = 2000
        pass
    return HttpResponse(JsonResponse({
        "code": code,
        "msg": "",
        "status": 1,
    }), content_type="application/json")

def adminEditUserInfo(request):
    user = None
    user_id = request.POST.get("id")
    user_account_name = request.POST.get("account_name")
    user_is_delete = request.POST.get('is_delete')
    user_type = request.POST.get('type')
    user_password = request.POST.get("password")
    user_real_name = request.POST.get("name")
    user_introduce = request.POST.get("introduce")
    user_email = request.POST.get("email")
    # print("user_id", user_id)
    code = 20000
    try:
        user = Users.objects.filter(account_name=user_account_name)
        if len(user) > 0:
            user = user.first()
            if user.id != int(user_id):
                # 账户名已存在
                code = 3000
                return HttpResponse(JsonResponse({
                    "code": code,
                    "msg": "",
                    "status": 1,
                }), content_type="application/json")

        user = Users.objects.filter(id=user_id).first()
        user.account_name = user_account_name
        if user_password != '':
            user.password = md5Encode(user_password)
        user.name = user_real_name
        user.introduce = user_introduce
        user.email = user_email
        if user_is_delete == '0' or user_is_delete == '1':
            user.is_delete = int(user_is_delete)
        if user_type == '1' or user_type == '2' or user_type == '3':
            user.type = int(user_type)

        user.save()
    except:
        code = 2000
        return HttpResponse(JsonResponse({
            "code": code,
            "msg": "",
            "status": 1,
        }), content_type="application/json")
    user = Users.objects.filter(id=user_id)
    return HttpResponse(JsonResponse({
        "code": code,
        "msg": "",
        "status": 1,
        "data": json.loads(serializers.serialize("json", user))
    }), content_type="application/json")

def getUserList(request):
    users = Users.objects.filter()
    for user in users:
        user.password = ''

    return HttpResponse(JsonResponse({
        "code": 20000,
        "msg": "",
        "status": 1,
        "data": json.loads(serializers.serialize("json", users))
    }), content_type="application/json")

def delUser(request):
    code = 20000
    try:
        user_id = request.GET.get('user_id')
        user = Users.objects.filter(id=user_id).first()
        user.is_delete = 1
        user.save()
    except:
        code = 2000

    return HttpResponse(JsonResponse({
        "code": code,
        "msg": "",
        "status": 1,
    }), content_type="application/json")

def addUser(request):
    register_account_name = request.POST.get("account_name")
    register_password = request.POST.get("password")
    register_email = request.POST.get("email")
    register_introduce = request.POST.get("introduce")
    register_type = request.POST.get("type")
    register_name = request.POST.get("name")
    try:
        register_password = md5Encode(register_password)
    except:
        pass
    user = Users.objects.filter(account_name=register_account_name)
    if len(user) != 0:
        user = user.first()
        if user.is_delete != 0:
            # 用户已删除，恢复该用户
            user.is_delete = 0
            user.account_name = register_account_name
            user.password = register_password
            user.email = register_email
            user.type = int(register_type)
            user.introduce = register_introduce
            user.name = register_name
            user.save()
            return HttpResponse(JsonResponse({
                "code": 20000,
                "msg": "用户注册成功！",
                "status": 1,
                "data": json.loads(serializers.serialize("json", [user]))
            }), content_type="application/json")
        else:
            # 用户未删除，已存在
            return HttpResponse(JsonResponse({
                "code": 2000,
                "msg": "用户已经存在！",
                "status": 1,
            }), content_type="application/json")
    else:
        user = Users.objects.create(name=register_account_name, account_name=register_account_name,
                                    password=register_password, email=register_email, introduce=register_introduce, is_delete=0, type=int(register_type))
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "用户注册成功！",
            "status": 1,
            "data": json.loads(serializers.serialize("json", [user]))
        }), content_type="application/json")

def doSignUp(request):
    register_account_name = request.POST.get("register_account_name")
    register_password = request.POST.get("register_password")
    register_email = request.POST.get("register_email")
    try:
        register_password = md5Encode(register_password)
    except:
        pass
    # print(password)

    user = Users.objects.filter(account_name=register_account_name)
    if len(user) != 0:
        user = user.first()
        if user.is_delete != 0:
            # 用户已删除，恢复该用户
            user.is_delete = 0
            user.account_name = register_account_name
            user.password = register_password
            user.email = register_email
            user.save()
            return HttpResponse(JsonResponse({
                "code": 20000,
                "msg": "用户注册成功！",
                "status": 1,
                "token": str(user.id),
            }), content_type="application/json")
        else:
            # 用户未删除，用户存在
            return HttpResponse(JsonResponse({
                "code": 2000,
                "msg": "用户已经存在！",
                "status": 1,
            }), content_type="application/json")
    else:
        user = Users.objects.create(name=register_account_name, account_name=register_account_name,
                                    password=register_password, email=register_email, introduce='', is_delete=0, type=1)
        return HttpResponse(JsonResponse({
            "code": 20000,
            "msg": "用户注册成功！",
            "status": 1,
            "token": str(user.id),
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
        temp['people_img'] = getBaseUrl() + resource.img
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


def getPeopleRelation(request):
    reloadPeopleImg()

    innerHTML_head = '<div class="c-my-node" style="background-image: url('
    innerHTML_mid = ');border:%s solid 3px;"><div class="c-node-name" style="color:%s">'
    innerHTML_tail = '</div></div>'

    video_id = request.POST.get('videoId')

    video = Videos.objects.filter(id=video_id).first()
    relation = {
        "rootId": video.title,
        "nodes": [
            {"id": video.title, "text": video.title, "color": '#ec6941', "borderColor": '#67C23A',
                'innerHTML': innerHTML_head + getBaseUrl() + video.snapshoot_img +
                ');border:%s solid 10px;"><div class="c-node-name" style="color:%s">' % (
                getRandomColor(), getRandomColor()) + video.title + innerHTML_tail
             },
        ],
        "links": []
    }
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg

    video_people = []
    appear_people_list = []
    if video.face_npy_path != None and video.face_npy_path != "" \
            and os.path.exists(os.path.join(curPath, video.face_npy_path)):
        loadData = np.load(os.path.join(curPath, video.face_npy_path)
                           , allow_pickle=True)
        # print(loadData)
        for item in loadData:
            new_item = {}
            new_item['people_name'] = item[0]
            peo = People.objects.filter(name = item[0]).first()
            new_item['people_id'] = peo.id
            appear_people_list.append(new_item['people_id'])
            if peo.img == None or peo == "":
                new_item['people_img'] = 'statics/resource/images/head_default.jpg'
            else:
                new_item['people_img'] = peo.img
            video_people.append(new_item)
        # print(video_people)

    for peo in video_people:
        # 添加节点
        node_item = {}
        node_item["id"] = peo["people_id"]
        node_item["text"] = peo["people_name"]
        node_item["borderColor"] = getRandomColor()
        node_item["color"] = getRandomColor()
        node_item["innerHTML"] = innerHTML_head + getBaseUrl() + peo["people_img"] + innerHTML_mid%(getRandomColor(), getRandomColor()) + peo["people_name"] + innerHTML_tail
        relation["nodes"].append(node_item)

        # 添加边
        relation_item = {}
        relation_item["from"] = video.title
        relation_item["to"] = str(peo["people_id"])
        relation_item["text"] = "出现"
        relation_item["color"] = getRandomColor()
        relation["links"].append(relation_item)

        # 遍历人物
        # from
        peo_relation_list = PeopleRelation.objects.filter(from_field=peo["people_id"])
        for rel in peo_relation_list:
            if rel.to not in appear_people_list:
                # 没出现过
                # 添加node
                p = People.objects.filter(id=rel.to).first()
                if p.img == None or p.img == "":
                    p.img = 'statics/resource/images/head_default.jpg'
                node_item = {}
                node_item["id"] = str(p.id)
                node_item["text"] = p.name
                node_item["borderColor"] = getRandomColor()
                node_item["color"] = getRandomColor()
                node_item["innerHTML"] = innerHTML_head + getBaseUrl() + p.img + innerHTML_mid%(getRandomColor(), getRandomColor()) + p.name + innerHTML_tail
                relation["nodes"].append(node_item)
                appear_people_list.append(p.id)
                # 添加边
                relation_item = {}
                relation_item["from"] = str(rel.from_field)
                relation_item["to"] = str(rel.to)
                relation_item["text"] = rel.text
                relation_item["color"] = getRandomColor()
                relation["links"].append(relation_item)
            else:
                # 是否已经存在改边
                i = 0
                while i < len(relation["links"]):
                    if relation["links"][i]["from"] == str(rel.from_field) and relation["links"][i]["to"] == str(rel.to):
                        break
                    else:
                        i += 1
                if i == len(relation["links"]):
                    # 添加边
                    relation_item = {}
                    relation_item["from"] = str(rel.from_field)
                    relation_item["to"] = str(rel.to)
                    relation_item["text"] = rel.text
                    relation_item["color"] = getRandomColor()
                    relation["links"].append(relation_item)

        # to
        peo_relation_list = PeopleRelation.objects.filter(to=peo["people_id"])
        for rel in peo_relation_list:
            if rel.from_field not in appear_people_list:
                # 没出现过
                # 添加node
                p = People.objects.filter(id=rel.from_field).first()
                if p.img == None or p.img == "":
                    p.img = 'statics/resource/images/head_default.jpg'
                node_item = {}
                node_item["id"] = str(p.id)
                node_item["text"] = p.name
                node_item["borderColor"] = getRandomColor()
                node_item["color"] = getRandomColor()
                node_item["innerHTML"] = innerHTML_head + getBaseUrl() + p.img + innerHTML_mid%(getRandomColor(), getRandomColor()) + p.name + innerHTML_tail
                relation["nodes"].append(node_item)
                appear_people_list.append(p.id)
                # 添加边
                relation_item = {}
                relation_item["from"] = str(rel.from_field)
                relation_item["to"] = str(rel.to)
                relation_item["text"] = rel.text
                relation_item["color"] = getRandomColor()
                relation["links"].append(relation_item)
            else:
                # 是否已经存在改边
                i = 0
                while i < len(relation["links"]):
                    if relation["links"][i]["from"] == str(rel.from_field) and relation["links"][i]["to"] == str(
                            rel.to):
                        break
                    else:
                        i += 1
                if i == len(relation["links"]):
                    # 添加边
                    relation_item = {}
                    relation_item["from"] = str(rel.from_field)
                    relation_item["to"] = str(rel.to)
                    relation_item["text"] = rel.text
                    relation_item["color"] = getRandomColor()
                    relation["links"].append(relation_item)


    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = relation
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getRandomColor():
    colors = ["#ec6941", "#6cc0ff", "#919926", "#ec9f41", "#79794f", "#9dd43e", "#45a01e", "#1cbf24"
              , "#1cbf9e", "#1c78bf", "#3234c2", "#6932c2", "#9732c2", "#dc3be8", "#e83b78"]
    index = random.randint(0, 14)
    return colors[index]


def reloadPeopleImg():
    people_list = People.objects.all()
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg

    for peo in people_list:
        if peo.img == None or peo.img == "" or not os.path.exists(os.path.join(curPath, peo.img)):
            # 添加头像
            imgs_path = os.path.join(curPath, 'demo_django', 'sq_face_recignition', 'train', peo.name.replace('.', ''))
            print(imgs_path)
            images = os.listdir(imgs_path)
            print(images)
            if len(images) > 0:
                # 开始添加头像
                img_path = os.path.join(imgs_path, images[0])
                head_db_path = 'statics/resource/head/' + peo.name.replace('.', '') + '.' + images[0].split('.')[-1]
                head_path = os.path.join(curPath, head_db_path)
                print(head_path)
                print(img_path)
                copyfile(img_path, head_path)
                peo.img = head_db_path
                peo.save()

def getPeopleRelationList(request):
    relation_list = []
    pr_list = PeopleRelation.objects.filter(is_delete=0)
    for pr in pr_list:
        item = {}
        item["id"] = pr.id
        item["from_field"] = pr.from_field
        p = People.objects.filter(id=str(pr.from_field)).first()
        item["from_people_name"] = p.name
        item["to"] = pr.to
        p = People.objects.filter(id=str(pr.to)).first()
        item["to_people_name"] = p.name
        item["text"] = pr.text
        relation_list.append(item)
    pass
    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    resp["data"] = relation_list
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def deletePeopleRealtion(request):
    delete_id = request.POST.get("delete_id")
    code = 20000
    try:
        pr = PeopleRelation.objects.filter(id=delete_id).first()
        if pr.is_delete == 0:
            pr.is_delete = 1
            pr.save()
    except:
        code = 2000
    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def addEditPeopleRelation(request):
    code = 20000
    resource = None
    try:
        dialog_status = request.POST.get('dialog_status')
        if dialog_status == 'add':
            #添加
            from_field = request.POST.get('from_field')
            to = request.POST.get('to')
            text = request.POST.get('text')
            list = PeopleRelation.objects.filter(from_field=from_field, to=to)
            if len(list) > 0:
                # 数据库中已经存在
                re = list.first()
                if re.is_delete != 0:
                    # 数据库中该记录被删除， 恢复更新内容
                    re.is_delete = 0
                    re.text = text
                    re.save()

            else:
                # 数据库中不存在
                pe = PeopleRelation.objects.create(from_field=from_field, to=to,
                                                            text=text, is_delete=0)

        elif dialog_status == 'edit':
            #修改
            id = request.POST.get('id')
            from_field = request.POST.get('from_field')
            to = request.POST.get('to')
            text = request.POST.get('text')
            pe = PeopleRelation.objects.filter(id=id).first()
            pe.from_field = int(from_field)
            pe.to = int(to)
            pe.text = text
            pe.save()
        else:
            code = 2000


    except:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getCollect(request):
    videoId = request.GET.get('videoId')
    user_id = request.GET.get('user_id')
    co = Collection.objects.filter(user_id=int(user_id), video_id=int(videoId))
    code = 20000
    if len(co) > 0:
        # 存在
        pass
    else:
        # 不存在
        code = 2000
        pass
    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def doCollect(request):
    videoId = request.GET.get('videoId')
    user_id = request.GET.get('user_id')
    co = Collection.objects.filter(user_id=int(user_id), video_id=int(videoId))
    code = 20000
    if len(co) > 0:
        # 存在则删除
        co = co.first()
        co.delete()
        code = 2000
    else:
        # 不存在则添加
        co = Collection.objects.create(user_id=int(user_id), video_id=int(videoId), time=int(time.time()))
        pass
    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getLikes(request):
    user_id = request.GET.get('user_id')
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    code = 20000
    co_list = Collection.objects.filter(user_id=user_id)
    videos = []
    for co in co_list:
        videoList = Videos.objects.filter(id=co.video_id)
        for video in videoList:
            # 没有缩略图会自动生成
            if video.snapshoot_img != None and video.snapshoot_img != "" and os.path.exists(
                    curPath + '/' + video.snapshoot_img):
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
        temp = {}
        temp['time'] = formate_time(int(co.time))
        temp['video_object'] = json.loads(serializers.serialize("json", videoList))
        videos.append(temp)
    # videos = serializers.serialize("json", videos)

    if len(videos) == 0:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["data"] = videos
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def clickVideo(request):
    code = 20000
    videoId = request.GET.get('videoId')
    user_id = request.GET.get('user_id')
    try:
        video = Videos.objects.filter(id=videoId).first()
        video.view_volume = video.view_volume + 1
        video.save()
        oper = Operations.objects.create(operation_type=1, user_id=user_id,
                                     video_id=videoId, operation_time=int(time.time()))
    except:
        code = 2000
    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def getViewHistory(request):
    user_id = request.GET.get('user_id')
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    code = 20000
    co_list = Operations.objects.filter(user_id=user_id).order_by("-operation_time")
    videos = []
    for co in co_list:
        videoList = Videos.objects.filter(id=co.video_id)
        for video in videoList:
            # 没有缩略图会自动生成
            if video.snapshoot_img != None and video.snapshoot_img != "" and os.path.exists(
                    curPath + '/' + video.snapshoot_img):
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
        temp = {}
        temp['time'] = formate_time(int(co.operation_time))
        temp['video_object'] = json.loads(serializers.serialize("json", videoList))
        videos.append(temp)
    # videos = serializers.serialize("json", videos)

    if len(videos) == 0:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["data"] = videos
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def addComment(request):
    videoId = request.POST.get('videoId')
    user_id = request.POST.get('user_id')
    comment = request.POST.get('comment')
    code = 20000
    comment_obj = None
    try:
        op = Operations.objects.create(operation_type=2, user_id=user_id,
                                video_id=videoId, comment=comment,
                                operation_time=int(time.time()))
        comment_obj = {}
        comment_obj['id'] = op.id
        comment_obj['user_id'] = op.user_id
        comment_obj['video_id'] = op.video_id
        comment_obj['content'] = op.comment
        comment_obj['createDate'] = formate_time(op.operation_time)
    except:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["data"] = comment_obj
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getCommentList(request):
    videoId = request.POST.get('videoId')
    code = 20000
    commentList = []
    try:
        op_list = Operations.objects.filter(operation_type=2, video_id=videoId).order_by('-operation_time')
        for op in op_list:
            user = Users.objects.filter(id=op.user_id).first()
            commentUser = {}
            commentUser['id'] = user.id
            commentUser['nickName'] = user.account_name
            commentUser['avatar'] = get_gravatar_url(user.account_name)
            comment = {}
            comment['id'] = op.id
            comment['content'] = op.comment
            comment['createDate'] = formate_time(op.operation_time)
            comment['commentUser'] = commentUser
            commentList.append(comment)
    except:
        code = 2000
    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["data"] = commentList
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getComments(request):
    user_id = request.GET.get('user_id')
    code = 20000
    commentList = []
    try:
        op_list = Operations.objects.filter(user_id=user_id, operation_type=2).order_by('-operation_time')
        for op in op_list:
            temp = {}
            video = Videos.objects.filter(id=op.video_id).first()
            temp['video_title'] = video.title
            temp['id'] = op.id
            temp['createDate'] = formate_time(op.operation_time)
            temp['content'] = op.comment
            commentList.append(temp)
    except:
        code = 2000
    print(commentList)
    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["data"] = commentList
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def get_gravatar_url(username, size=80):
    '''返回头像url'''
    styles = ['identicon', 'monsterid', 'wavatar', 'retro']
    '''
    mm： 简约、卡通风格的人物轮廓像（不会随邮箱哈希值变化而变化）。
    identicon：几何图案，其形状会随电子邮箱哈希值变化而变化。
    monsterid：程序生成的“怪兽”头像，颜色和面孔会随会随电子邮箱哈希值变化而变化。
    wavatar:：用不同面容和背景组合生成的面孔头像。
    retro：程序生成的8位街机像素头像。
    '''
    m5 = hashlib.md5(f'{username}'.encode('utf-8')).hexdigest()  # 返回16进制摘要字符串
    url = f'http://www.gravatar.com/avatar/{m5}?s={size}&d=wavatar'  # s 返回头像大小 d 返回头像类型 没在gravatar.com 注册的邮箱需要加此参数
    return url

def doVideoImg():
    pass


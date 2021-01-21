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
from django.core import serializers
import numpy as np
from django.http import HttpResponse, JsonResponse, response
from django.shortcuts import render, redirect
from MyModel.models import Users, Videos, Binner, People, SourceInformation, Country
import hashlib

from demo_django.asr.pyTranscriber.asr_api import asr_subtitle
from demo_django.utils import getBaseUrl, md5Encode

import re
from wsgiref.util import FileWrapper
from django.http import StreamingHttpResponse

# zzw
from demo_django.wiki.wiki import wiki_api

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
    content_type = content_type or 'application/octet-stream'
    if range_match:
        first_byte, last_byte = range_match.groups()
        first_byte = int(first_byte) if first_byte else 0
        last_byte = first_byte + 1024 * 1024 * 8  # 8M 每片,响应体最大体积
        if last_byte >= size:
            last_byte = size - 1
        length = last_byte - first_byte + 1
        resp = StreamingHttpResponse(file_iterator(path, offset=first_byte, length=length), status=206,
                                     content_type=content_type)
        resp['Content-Length'] = str(length)
        resp['Content-Range'] = 'bytes %s-%s/%s' % (first_byte, last_byte, size)
    else:
        # 不是以视频流方式的获取时，以生成器方式返回整个文件，节省内存
        resp = StreamingHttpResponse(FileWrapper(open(path, 'rb')), content_type=content_type)
        resp['Content-Length'] = str(size)
    resp['Accept-Ranges'] = 'bytes'
    return resp

def streamVideo(request):
    videoId = request.GET.get("videoId")
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    path = os.path.join(curPath, video.rel_path)
    return stream_video(request, path)

def getAllVideos(request):
    video_lists = Videos.objects.all()
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    for video in video_lists:
        # 没有缩略图会自动生成
        if video.snapshoot_img != None and video.snapshoot_img != "":
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

        video.rel_path = getBaseUrl() + video.rel_path

        if video.text_location != None:
            video.text_location = getBaseUrl() + video.text_location
        if video.subtitle != None:
            video.subtitle = getBaseUrl() + video.subtitle
        if video.asr_path != None:
            video.asr_path = getBaseUrl() + video.asr_path
    videos = serializers.serialize("json", video_lists)


    return HttpResponse(JsonResponse({
        "code": 0,
        "msg": "",
        "status": 1,
        "data": json.loads(videos)
    }), content_type="application/json")

def getSubTitle(request):
    videoId = request.GET.get("videoId")
    print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 字幕
    subTitle = []
    if video.subtitle != None and video.subtitle != "":
        print("subtitle", os.path.join(curPath, video.subtitle))
        with open(os.path.join(curPath, video.subtitle), 'r') as f:
            line = f.readline()
            while line:
                # print("line", line)
                item = {}
                if "-->" in line:
                    # print(line)
                    item["time"] = line
                    line = f.readline()
                    item["content"] = line
                    line = f.readline()
                if item:
                    subTitle.append(item)
                line = f.readline()
    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["subTitle"] = subTitle
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getVideoAdditionData(request):

    videoId = request.GET.get("videoId")
    print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg



    # 附加信息
    addition_data = {}
    video_time = get_video_times(os.path.join(curPath, video.rel_path))
    addition_data["video_time"] = video_time

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

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["addition_data"] = addition_data
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getVideoEquipment(request):
    videoId = request.GET.get("videoId")
    print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    equipment_json_data = None
    if video.equipment_json_path != None and video.equipment_json_path != "":
        with open(os.path.join(curPath, video.equipment_json_path), 'r') as load_f:
            load_dict = json.load(load_f)

            print(load_dict)
            for item in load_dict:
                print(item)
                item["filename"] = getBaseUrl() + "statics/resource/object_images/" + item["filename"]
            equipment_json_data = load_dict

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["equipment_json_data"] = equipment_json_data
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def getVideoPPT(request):
    videoId = request.GET.get("videoId")
    print(videoId)
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

def getFace(request):
    videoId = request.GET.get("videoId")
    print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 人脸识别
    face_data = None
    if video.face_npy_path != None and video.face_npy_path != "":
        loadData = np.load(os.path.join(curPath, video.face_npy_path)
                           , allow_pickle=True)
        face_data = []
        for item in loadData:
            print(item)
            item_item = {"name": item[0]}
            item_list = []
            for i in range(len(item[1])):
                item[2][i] = getBaseUrl() + "statics/" + "resource/" + "face_images/" + item[2][i]
                item_list.append({
                    'time': item[1][i],
                    'img': item[2][i]
                })
            print(item_list)
            item_item["time_img"] = item_list

            # 人物介绍
            print("name:", item_item["name"])
            people = People.objects.filter(name=item_item["name"]).first()
            if people != None:
                if people.introduce == None or people.introduce == "":
                    introduce = wiki_api(people.name)
                    if introduce != None and introduce != "":
                        item_item["introduce"] = introduce
                        people.introduce = introduce
                        people.save()
                else:
                    item_item["introduce"] = people.introduce

                item_item["head_img"] = None
                if people.img != None and people.img != "":
                    item_item["head_img"] = getBaseUrl() + people.img
            else:
                item_item["introduce"] = None
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
    print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    # 文本识别
    text_data = None
    if video.text_location != None and video.text_location != "":
        with open(os.path.join(curPath, video.text_location), 'r') as load_f:
            text_data = json.load(load_f)
        print(text_data)

    resp = {}
    resp["code"] = 0
    resp["msg"] = ""
    resp["status"] = 1
    resp["text_data"] = text_data
    return HttpResponse(JsonResponse(resp), content_type="application/json")



def getOneVideos(request):

    videoId = request.GET.get("videoId")
    print(videoId)
    video = Videos.objects.filter(id=videoId).first()

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg


    video.rel_path = getBaseUrl() + video.rel_path

    if video.subtitle != None and video.subtitle != "":
        video.subtitle = getBaseUrl() + video.subtitle
    if video.asr_path != None and video.asr_path != "":
        video.asr_path = getBaseUrl() + video.asr_path
    if video.ppt_pdf_path != None and video.ppt_pdf_path != "":
        video.ppt_pdf_path = getBaseUrl() + video.ppt_pdf_path

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
        print(item)
        item_item = {"name": item[0]}
        item_list = []
        for i in range(len(item[1])):
            item[2][i] = getBaseUrl()+"statics/"+"resource/"+"images/"+item[2][i]
            item_list.append({
                'time': item[1][i],
                'img': item[2][i]
            })
        print(item_list)
        item_item["time_img"] = item_list
        data.append(item_item)
    res = {
        "code": 0,
        "msg": "",
        "status": 1,
        "data": data
    }
    print(res)
    return HttpResponse(JsonResponse(res), content_type="application/json")


def getText(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    with open(os.path.join(curPath, 'statics', 'resource', 'text', 'test.json'), 'r') as load_f:
        load_dict = json.load(load_f)
    # print(load_dict)
    res = {
        "code": 0,
        "msg": "",
        "status": 1,
        "data": load_dict
    }
    return HttpResponse(JsonResponse(res), content_type="application/json")






# 获得某一视频中出现的装备及出现时刻
def getEquipment(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    with open(os.path.join(curPath, 'statics', 'resource', 'text', 'result6.json'), 'r') as load_f:
        load_dict = json.load(load_f)
    print(load_dict)
    res = {
        "code": 0,
        "msg": "",
        "status": 1,
        "data": load_dict
    }
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
    account_name = request.GET.get("account_name")
    password = request.GET.get("password")
    password = md5Encode(password)
    print(account_name+password)
    user = Users.objects.filter(account_name=account_name, password=password)
    print(user)
    if len(user) != 0:
        user = serializers.serialize("json", user)
        return HttpResponse(JsonResponse({
            "code": 0,
            "msg": "成功查找到该用户，该用户存在！",
            "status": 1,
            "data": json.loads(user)
        }), content_type="application/json")
    else:
        return HttpResponse(JsonResponse({
            "code": 1,
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
    print(password)

    user = Users.objects.filter(name=name)
    if len(user) != 0:
        return HttpResponse(JsonResponse({
            "code": 1,
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
            "code": 0,
            "msg": "用户注册成功！",
            "status": 1,
            "data": {
                "name": name,
                "account_name": account_name
            }
        }), content_type="application/json")


def uploadvideo(request):
        file = request.FILES.get("file")
        title = request.POST.get("title")
        print("title"+title)
        print("aa---------------------------------进入！")

        curPath = os.path.abspath(os.path.dirname(__file__))
        split_reg = curPath.split(os.sep)[-1]
        curPath = curPath.split(split_reg)[0]+split_reg
        print(curPath)

        file_name = str(uuid.uuid1()) + "." + str(file.name).split('.')[-1]
        video_path = os.path.abspath(os.path.join(curPath, 'statics', 'resource', 'videos', file_name))
        video_path_db = 'statics/'+'resource/'+'videos/'+file_name
        f = open(video_path, 'wb')
        for i in file.chunks():
            f.write(i)
        f.close()

        # asr 语音识别，生成字幕文件
        asr_subtitle(video_path, os.path.join(curPath, 'statics', 'resource', 'audio_text'))
        asr_path = 'statics/'+'resource/'+'audio_text/'+file_name.split(".")[0]+".txt"
        subtitle = 'statics/'+'resource/'+'audio_text/'+file_name.split(".")[0]+".srt"

        # 抽帧作为缩略图
        image_path = os.path.join(curPath, 'statics', 'resource', 'images', file_name.split(".")[0]+".png")
        image_path_db = 'statics/'+'resource/'+'images/'+file_name.split(".")[0]+".png"
        extract_image(video_path, image_path=image_path)


        v = Videos.objects.create(name=file_name, rel_path=video_path_db, introduce=file_name,
                                    create_time=int(time.time()),
                                    tag=file_name, title=title, asr_path=asr_path,
                                    subtitle=subtitle, snapshoot_img=image_path_db
                                    )
        return JsonResponse({
            "code": 200,
            "msg": "",
            "status": 200,
            "data": {
                "filename": file_name,
                "fileid": v.id,
            }
        })

def extract_image(video_path, image_path):
    cap = cv2.VideoCapture(video_path)  # 读入文件
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)  # 从num帧开始读视频
    success, frame = cap.read()
    cv2.imwrite(image_path, frame)
    cap.release()
    print("抽帧成功！")

def myTest(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)

    # tz = pytz.timezone('Asia/Shanghai')
    # time_upload = datetime.now(tz)
    t = int(time.time())
    print(t)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)))
    return JsonResponse({
        "code": 0,
        "msg": "",
        "status": 1,
        "fileid": 0,
        "data": {}
    })


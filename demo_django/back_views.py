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

# from demo_django.AdelaiDet.TextRecognize_API import text_recognize
from demo_django.asr.KDXF_ASR.asr_api import asr_subtitle_kdxf
from demo_django.asr.pyTranscriber.asr_api import asr_subtitle
# from demo_django.darknet.ObjectDetection_API import objectDetection
from demo_django.ppt.ppt_api import ppt_api
from demo_django.product.OCR_report import ocr_report_return_with_path
from demo_django.sq_face_recignition.FaceRecognition_API import face_recognition
from demo_django.translate.translate import translate_api
from demo_django.utils import getBaseUrl, md5Encode

import re
from wsgiref.util import FileWrapper
from django.http import StreamingHttpResponse

# zzw
from demo_django.views import extract_image, formate_time, parseVTTForUpdate
# from demo_django.voice.test_api import recognition_video
# from demo_django.voice.test_api import recognition_video
from demo_django.voice.test_api import recognition_video_api
from demo_django.wiki.wiki import wiki_api




def uploadvideo(request):
    file = request.FILES.get("file")
    title = request.POST.get("title")
    source = request.POST.get("source")
    functions = request.POST.get("functions")
    tag = request.POST.get("tag")
    print("function", functions)
    print("title" + title)
    print("aa---------------------------------进入！")

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    # print(curPath)

    # 保存视频文件
    file_name = str(uuid.uuid1()) + "." + str(file.name).split('.')[-1]
    video_path = os.path.abspath(os.path.join(curPath, 'statics', 'resource', 'videos', file_name))
    video_path_db = 'statics/' + 'resource/' + 'videos/' + file_name
    f = open(video_path, 'wb')
    for i in file.chunks():
        f.write(i)
    f.close()
    print("保存视频文件")

    s = SourceInformation.objects.filter(name = source).first()

    if s != None:
        source_id = s.id
    else:
        source_id = None

    video = Videos.objects.create(name=file_name, rel_path=video_path_db, introduce=file_name,
                              create_time=int(time.time()),
                              tag=tag, title=title,
                              source_id=source_id, is_delete=0, view_volume=0
                              )


    image_path_db = None
    try:
        # 抽帧作为缩略图
        image_path = os.path.join(curPath, 'statics', 'resource', 'images', file_name.split(".")[0] + ".png")
        image_path_db = 'statics/' + 'resource/' + 'images/' + file_name.split(".")[0] + ".png"
        extract_image(video_path, image_path=image_path)
    except:
        image_path_db = None
        pass
    if image_path_db != None:
        video.image_path_db = image_path_db
        video.save()
        print("抽帧作为缩略图")


    # try:
    #     if '1' in functions:
    #         # object detection
    #         video_path = curPath + '/' + video.rel_path
    #         result_video_with_border_return, outputjson_return = objectDetection(video_path)
    #         video.border_video_path = result_video_with_border_return
    #         video.equipment_json_path = outputjson_return
    #         video.save()
    #         print("object detection")
    # except:
    #     pass

    try:
        if '2' in functions:
            # PPT detection
            video_path = curPath + '/' + video.rel_path
            json_file_path, pdf_path = ppt_api(video_path)
            video.ppt_json_path = json_file_path
            video.ppt_pdf_path = pdf_path
            video.save()
            print("PPT detection")
    except:
        pass

    # try:
    #     if '3' in functions:
    #         # ocr detection
    #         video_path = curPath + '/' + video.rel_path
    #         file_name = text_recognize(video_path)
    #         video.text_location = file_name
    #         video.save()
    #         print("ocr detection")
    #
    #         ocr_pdf_path = ocr_report_return_with_path(curPath + '/' + video.text_location)
    #         if ocr_pdf_path != None:
    #             video.text_pdf_location = ocr_pdf_path
    #             video.save()
    #             print("ocr pdf done")
    # except:
    #     pass


    try:
        if '0' in functions:
            # face detection
            video_path = curPath + '/' + video.rel_path
            print("chuli")
            output_npy_db = face_recognition(video_path)
            video.face_npy_path = output_npy_db
            video.save()
            print("face detection")
    except:
        pass

    try:

        if '4' in functions:
            # asr 语音识别，生成字幕文件
            # asr_subtitle(video_path, os.path.join(curPath, 'statics', 'resource', 'audio_text'))
            asr_subtitle_kdxf(video_path, os.path.join(curPath, 'statics', 'resource', 'audio_text'))
            asr_path = 'statics/' + 'resource/' + 'audio_text/' + file_name.split(".")[0] + ".txt"
            subtitle = 'statics/' + 'resource/' + 'audio_text/' + file_name.split(".")[0] + ".srt"
            video.subtitle = subtitle
            video.asr_path = asr_path
            video.save()
            print("asr 语音识别，生成字幕文件")
            if video.translate_subtitle == None or video.translate_subtitle == "":
                asr_translate_path_db, subtitle_translate_path_db = translate_file_en_zh(curPath + '/' + video.subtitle)
                video.translate_asr_path = asr_translate_path_db
                video.translate_subtitle = subtitle_translate_path_db
                video.save()
                print("translate end")
    except:
        pass

    try:
        if '5' in functions:
            # voice detection
            pass
    except:
        pass


    return JsonResponse({
        "code": 20000,
        "msg": "",
        "status": 200,
    })

def translate_file_en_zh(file_path):

    num_list = []
    time_list = []
    content_list = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            print("line", line)
            num_list.append(line)
            line = f.readline()
            if "-->" in line:
                print(line)
                time_list.append(line)
                line = f.readline()
                content_list.append(line)
                line = f.readline()
            line = f.readline()
    print("num_list", num_list)
    print("time_list", time_list)
    print("content_list", content_list)
    for i in range(len(content_list)):
        time.sleep(1.0)
        translate_content = ""
        try:
            translate_content = translate_api(content_list[i])
            print("translate_content", translate_content)
        except:
            translate_content = ""
        if translate_content == None:
            translate_content = ''
        content_list[i] += translate_content + '\n'
    print("num_list", num_list)
    print("time_list", time_list)
    print("content_list", content_list)

    file_name = file_path.split('/')[-1].split('.')[0]
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    # print(curPath)

    asr_translate_path_db = 'statics/resource/audio_text/' + file_name + '_translate.txt'
    asr_translate_path = curPath + '/' + asr_translate_path_db

    subtitle_translate_path_db = 'statics/resource/audio_text/' + file_name + '_translate.srt'
    subtitle_translate_path = curPath + '/' + subtitle_translate_path_db

    with open(asr_translate_path, 'w') as f:
        for content in content_list:
            f.write(content)
    with open(subtitle_translate_path, 'w') as f:
        for i in range(len(content_list)):
            f.write(num_list[i])
            f.write(time_list[i])
            f.write(content_list[i])
            f.write('\n')
    return asr_translate_path_db, subtitle_translate_path_db


def srt2vtt(file_name):
    content = open(file_name, "r", encoding="utf-8").read()

    # 添加WEBVTT行
    content = "WEBVTT\n\n" + content

    # 替换“,”为“.”
    content = re.sub("(\d{2}:\d{2}:\d{2}),(\d{3})", lambda m: m.group(1) + '.' + m.group(2), content)
    # output_file = os.path.splitext(file_name)[0] + '.vtt'

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    # print(curPath)

    output_filename_db = 'statics/resource/audio_text/' + file_name.split('/')[-1].split('.')[0] + '.vtt'
    output_filename = curPath + '/' + output_filename_db
    open(output_filename, "w", encoding="utf-8").write(content)
    return output_filename_db



def deletevideo(request):
    videoId = request.GET.get("videoId")
    video = Videos.objects.filter(id=videoId).first()
    video.is_delete = 1
    video.save()
    return JsonResponse({
        "code": 200,
        "msg": "",
        "status": 200,
    })

def updateSubTitleItem(request):

    videoId = request.POST.get("videoId")
    update_index = request.POST.get("update_index")
    update_time = request.POST.get("update_time")[1:-1]
    update_content = json.loads(request.POST.get("update_content"))

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    video = Videos.objects.filter(id=videoId).first()

    sub_obj = None
    if video.translate_subtitle != None and video.translate_subtitle != ''\
            and os.path.exists(os.path.join(curPath, video.translate_subtitle)):
        sub_obj = parseVTTForUpdate(os.path.join(curPath, video.translate_subtitle))
    code = 20000
    if sub_obj:
        # print(sub_obj[int(update_index)]['time'], update_time[1:-1])
        if str(sub_obj[int(update_index)]['time']) == update_time:
            sub_obj[int(update_index)]['content'] = update_content
            srt_content = ''
            for i in range(len(sub_obj)):
                srt_content += str(i+1) + '\n'
                srt_content += sub_obj[i]['time'] + '\n'
                for ii in range(len(sub_obj[i]['content'])):
                    srt_content += sub_obj[i]['content'][ii] + '\n'
                srt_content += '\n'

            with open(curPath + '/' + video.translate_subtitle, 'w', encoding='utf-8') as wf:
                wf.write("WEBVTT\n\n" + srt_content)

        else:
            code = 2000
    else:
        code = 2000

    resp = {}
    resp["code"] = code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def updateSubTitle(request):
    videoId = request.POST.get("videoId")

    sub_obj = json.loads(request.POST.get('sub_obj'))
    print('sub_obj', sub_obj)

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    video = Videos.objects.filter(id=videoId).first()

    srt_content = ''
    for i in range(len(sub_obj)):
        srt_content += str(i+1) + '\n'
        srt_content += sub_obj[i]['time'] + '\n'
        for ii in range(len(sub_obj[i]['content'])):
            srt_content += sub_obj[i]['content'][ii] + '\n'
        srt_content += '\n'

    with open(curPath + '/' + video.translate_subtitle, 'w', encoding='utf-8') as wf:
        wf.write("WEBVTT\n\n" + srt_content)

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def updateOCRItem(request):
    videoId = request.POST.get("videoId")
    deleteIndex = request.POST.get("deleteIndex")
    updateContent = request.POST.get('updateContent')

    print(videoId, deleteIndex, updateContent)
    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"
    return_code = 20000
    try:
        text_data = None
        if video.text_location != None and video.text_location != '':
            with open(os.path.join(curPath, video.text_location), 'r') as load_f:
                text_data = json.load(load_f)
            print(text_data)
        if text_data != None:
            text_data[int(deleteIndex)]['content'] = updateContent
            with open(os.path.join(curPath, video.text_location), 'w') as load_f:
                json.dump(text_data, load_f)
    except:
        return_code = 2000
        pass
    resp = {}
    resp["code"] = return_code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def delOCRItem(request):
    videoId = request.POST.get("videoId")
    deleteIndex = request.POST.get("deleteIndex")
    # print(videoId, deleteIndex)
    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"
    return_code = 20000
    try:
        text_data = None
        if video.text_location != None and video.text_location != '':
            with open(os.path.join(curPath, video.text_location), 'r') as load_f:
                text_data = json.load(load_f)
            print(text_data)
        if text_data != None:
            text_data.remove(text_data[int(deleteIndex)])
            with open(os.path.join(curPath, video.text_location), 'w') as load_f:
                json.dump(text_data, load_f)
    except:
        return_code = 2000
        pass
    resp = {}
    resp["code"] = return_code
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def updateOCR(request):
    videoId = request.POST.get("videoId")
    updateTime = request.POST.get("updateTime").split('^$')
    updateOCR = request.POST.get("updateOCR").split('^$')

    video = Videos.objects.filter(id=videoId).first()

    print("updateTime", updateTime)
    print("updateOCR", updateOCR)

    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"

    text_data = None
    if video.text_location != None and video.text_location != '':
        with open(os.path.join(curPath, video.text_location), 'r') as load_f:
            text_data = json.load(load_f)
        # print(text_data)

    if text_data != None:
        for i in range(len(text_data)):
            # print(text_data[i]['time'], updateTime[i])
            if text_data[i]['time'] == updateTime[i][1:-1]:
                # print("ok")
                text_data[i]['content'] = updateOCR[i]

    if video.text_location != None and video.text_location != '':
        with open(os.path.join(curPath, video.text_location), 'w') as load_f:
            json.dump(text_data, load_f)
        # print("text_data", type(text_data))
        # print("text_data", text_data)

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")


def updatePPT(request):
    videoId = request.POST.get("videoId")
    updateImages = request.POST.get("updateImages").split('^$')

    print("updateImages", updateImages)
    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"

    result_dict = {}
    if video.ppt_json_path != None and video.ppt_json_path != ""\
            and os.path.exists(os.path.join(curPath, video.ppt_json_path)):
        for i in range(len(updateImages)):
            result_dict[str(i)] = updateImages[i].split('ppt_images')[-1][1:]
        with open(curPath + '/' + video.ppt_json_path, 'w') as load_f:
            json.dump(result_dict, load_f)

    print("result_dict", result_dict)

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")

def delPPTImg(request):
    # pass
    videoId = request.POST.get("videoId")
    del_index = request.POST.get("del_index")

    video = Videos.objects.filter(id=videoId).first()
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"

    ppt_dict = {}
    ppts = []
    if video.ppt_json_path != None and video.ppt_json_path != ""\
            and os.path.exists(os.path.join(curPath, video.ppt_json_path)):
        with open(curPath + '/' + video.ppt_json_path, 'r') as load_f:
            ppt_dict = json.load(load_f)

    # print("ppt_dict", ppt_dict)

    for item in ppt_dict:
        ppts.append(ppt_dict[item])
    ppts.remove(ppts[int(del_index)])
    # print(ppts)
    result_dict = {}
    for i in range(len(ppts)):
        result_dict[str(i)] = ppts[i]

    with open(curPath + '/' + video.ppt_json_path, 'w') as load_f:
        json.dump(result_dict, load_f)

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")



def searchVideos(request):
    searchType = request.POST.get("searchType")
    searchText = request.POST.get("searchText")
    searchText = str(searchText).upper()
    print("searchText", searchText)

    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"

    video_lists = Videos.objects.filter(is_delete=0)
    video_lists = list(video_lists)
    print("type(video_lists_list)", len(video_lists))

    if '0' in searchType:
        # latest upload
        # video_i = 0
        # while video_i < len(video_lists):
        #     video_type = str(video_lists[video_i].name.split('.')[-1]).upper()
        #     if video_type == None or video_type == "" or searchText.split('.')[-1] != video_type:
        #         video_lists.remove(video_lists[video_i])
        #         video_i -= 1
        #     video_i += 1

        video_lists = Videos.objects.filter(is_delete=0).order_by("-create_time")
        video_lists = list(video_lists)
        pass
    if '1' in searchType:
        # people face
        video_i = 0
        while video_i < len(video_lists):
            if video_lists[video_i].face_npy_path != None and video_lists[video_i].face_npy_path != "" \
                    and os.path.exists(os.path.join(curPath, video_lists[video_i].face_npy_path)):
                video_face_path = curPath + '/' + str(video_lists[video_i].face_npy_path)
                text_data = np.load(video_face_path, allow_pickle=True)
                text_data = str(text_data).upper()
                # print("text_data", text_data)
                if searchText not in text_data:
                    # print("shanchu")
                    video_lists.remove(video_lists[video_i])
                    video_i -= 1
            else:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        # print(len(video_lists))
        pass
    if '2' in searchType:
        # object
        # print("searchText", searchText)
        name_dict = {}
        name_dict['直升机'] = 'helicopter'
        name_dict['水面舰艇'] = 'military ship'
        name_dict['装甲车'] = 'armored car'
        name_dict['战斗机'] = 'fighter'
        searchText1 = searchText
        try:

            searchText = name_dict[searchText]
        except:
            pass

        if searchText != None:
            # searchText = searchText.upper()
            # print("searchText", searchText)
            video_i = 0
            while video_i < len(video_lists):
                if video_lists[video_i].equipment_json_path != None and video_lists[video_i].equipment_json_path != ""\
                        and os.path.exists(os.path.join(curPath, video_lists[video_i].equipment_json_path)):
                    video_equipment_path = curPath + '/' + str(video_lists[video_i].equipment_json_path)
                    text_data = None
                    with open(video_equipment_path, 'r') as f:
                        text_data = json.load(f)
                    if text_data != None:
                        text_data = str(text_data).upper()
                        # print("text_data", text_data)
                        if searchText not in text_data and searchText1 not in text_data:
                            video_lists.remove(video_lists[video_i])
                            video_i -= 1
                    else:
                        video_lists.remove(video_lists[video_i])
                        video_i -= 1
                else:
                    video_lists.remove(video_lists[video_i])
                    video_i -= 1
                video_i += 1
        else:
            video_lists = []
        pass
    if '3' in searchType:
        # introduce
        video_i = 0
        while video_i < len(video_lists):
            video_introduce = str(video_lists[video_i].introduce).upper()
            if video_introduce == None or video_introduce == "" or searchText not in video_introduce:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        pass
    if '4' in searchType:
        # source
        video_i = 0
        while video_i < len(video_lists):
            video_source = None
            if video_lists[video_i].source_id != None:
                source = SourceInformation.objects.filter(id=video_lists[video_i].source_id).first()
                if source:
                    video_source = str(source.name).upper()

            if video_source == None or searchText not in video_source:
                video_lists.remove(video_lists[video_i])
                video_i -= 1

            video_i += 1
        # print("video_lists", len(video_lists))
        pass
    if '5' in searchType:
        # tag
        video_i = 0
        while video_i < len(video_lists):
            video_tag = str(video_lists[video_i].tag).upper()
            if video_tag == None or video_tag == "" or searchText not in video_tag:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        pass
    if '6' in searchType:
        # id
        video_lists = Videos.objects.filter(is_delete=0, id=searchText)
        video_lists = list(video_lists)
        pass
    if '7' in searchType:
        # ocr content
        video_i = 0
        while video_i < len(video_lists):
            if video_lists[video_i].text_location != None and video_lists[video_i].text_location != "":
                video_text_path = curPath + '/' + str(video_lists[video_i].text_location)
                text_data = None
                with open(video_text_path, 'r') as f:
                    text_data = f.readlines()
                if text_data != None:
                    text_data = str(text_data).upper()
                    if searchText not in text_data:
                        video_lists.remove(video_lists[video_i])
                        video_i -= 1
                else:
                    video_lists.remove(video_lists[video_i])
                    video_i -= 1
            else:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        pass
    if '8' in searchType:
        # srt vtt
        video_i = 0
        while video_i < len(video_lists):
            if video_lists[video_i].translate_subtitle != None and video_lists[video_i].translate_subtitle != "":
                video_subtitle_path = curPath + '/' + str(video_lists[video_i].translate_subtitle)
                text_data = None
                with open(video_subtitle_path, 'r', encoding="utf-8") as f:
                    text_data = f.readlines()
                if text_data != None:
                    text_data = str(text_data).upper()
                    # print("text_data", type(text_data))
                    if searchText not in text_data:
                        video_lists.remove(video_lists[video_i])
                        video_i -= 1
                else:
                    video_lists.remove(video_lists[video_i])
                    video_i -= 1
            else:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        pass
    if '9' in searchType:
        # country
        video_i = 0
        while video_i < len(video_lists):
            # print("len(video_lists)", len(video_lists))
            # print("video_i", video_i)
            video_country = None
            if video_lists[video_i].country_id != None:
                country = Country.objects.filter(id=video_lists[video_i].country_id).first()
                if country:
                    video_country = country.name
            if video_country == None or searchText not in video_country:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        pass
    if 'a' in searchType:
        # title
        video_i = 0
        while video_i < len(video_lists):
            video_title = str(video_lists[video_i].title).upper()
            if video_title == None or video_title == "" or searchText not in video_title:
                video_lists.remove(video_lists[video_i])
                video_i -= 1
            video_i += 1
        pass

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
        video.create_time = formate_time(video.create_time)

        if video.text_location != None:
            video.text_location = getBaseUrl() + video.text_location
        if video.subtitle != None:
            video.subtitle = getBaseUrl() + video.subtitle
        if video.asr_path != None:
            video.asr_path = getBaseUrl() + video.asr_path
    videos = serializers.serialize("json", video_lists)

    return HttpResponse(JsonResponse({
        "code": 20000,
        "msg": "",
        "status": 1,
        "data": json.loads(videos)
    }), content_type="application/json")



def deleteEquipment(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    videoId = request.POST.get('videoId')
    imageIndex = request.POST.get('imageIndex')
    equipmentIndex = request.POST.get('equipmentIndex')

    print("videoId", videoId)
    print("imageIndex", imageIndex)
    print("equipmentIndex", equipmentIndex)

    video = Videos.objects.filter(id=videoId).first()
    if video.equipment_json_path != None and video.equipment_json_path != "" and os.path.exists(
            os.path.join(curPath, video.equipment_json_path)):
        with open(os.path.join(curPath, video.equipment_json_path), 'r') as load_f:
            load_dict = json.load(load_f)
            del load_dict[int(imageIndex)]['objects'][int(equipmentIndex)]
            if len(load_dict[int(imageIndex)]['objects']) == 0:
                del load_dict[int(imageIndex)]
        with open(os.path.join(curPath, video.equipment_json_path), 'w') as load_f:
            json.dump(load_dict, load_f)
    pass

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")



def updateEquipment(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    videoId = request.POST.get('videoId')
    imageIndex = request.POST.get('imageIndex')
    equipmentIndex = request.POST.get('equipmentIndex')
    equipmentName = request.POST.get('equipmentName')

    print("videoId", videoId)
    print("imageIndex", imageIndex)
    print("equipmentIndex", equipmentIndex)
    print("equipmentName", equipmentName)

    video = Videos.objects.filter(id=videoId).first()
    if video.equipment_json_path != None and video.equipment_json_path != "" and os.path.exists(os.path.join(curPath, video.equipment_json_path)):
        with open(os.path.join(curPath, video.equipment_json_path), 'r') as load_f:
            load_dict = json.load(load_f)
            load_dict[int(imageIndex)]['objects'][int(equipmentIndex)]['name'] = equipmentName
        with open(os.path.join(curPath, video.equipment_json_path), 'w') as load_f:
            json.dump(load_dict, load_f)
    pass

    resp = {}
    resp["code"] = 20000
    resp["msg"] = ""
    resp["status"] = 1
    return HttpResponse(JsonResponse(resp), content_type="application/json")




def updateUserInfo(request):
    account_name = request.POST.get("account_name")
    password = request.POST.get("password")
    type = request.POST.get("type")
    # user = Users.objects.filter(account_name=account_name).first()

    return HttpResponse(JsonResponse({
        "code": 0,
        "msg": "",
        "status": 1,
        "data": None
    }), content_type="application/json")


def changeVideoInfo(request):
    videoId = request.POST.get("videoId")
    # print(videoId)
    new_value = request.POST.get('new_value')
    # print(new_value)
    index = int(request.POST.get('index'))
    # print(index)

    video = Videos.objects.filter(id=videoId).first()
    if index == 0:
        #修改视频来源
        video.source_id = int(new_value)
        video.save()
        pass
    elif index == 1:
        #修改视频介绍
        video.introduce = new_value
        video.save()
        pass
    elif index == 2:
        # 修改来源国家
        video.country_id = int(new_value)
        video.save()
        pass
    elif index == 3:
        tagList = json.loads(new_value)
        # print(tagList, type(tagList))
        tag_str = ''
        for item in tagList:
            tag_str += item + ' '
        tag_str = tag_str.strip()
        video.tag = tag_str
        video.save()
        pass
    return HttpResponse(JsonResponse({
        "code": 20000,
        "msg": "",
        "status": 1,
    }), content_type="application/json")





def myTest(request):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = curPath.split(os.sep)[-1]
    curPath = curPath.split(split_reg)[0] + split_reg

    videos = Videos.objects.all()
    # videos = list(videos)
    print('videos', videos)

    for video in videos:
        print('video', video)
        if video.subtitle != None and video.subtitle != '':
            asr_translate_path_db, subtitle_translate_path_db = translate_file_en_zh(curPath + '/' + video.subtitle)
            print("subtitle_translate_path_db", subtitle_translate_path_db)
            video.translate_asr_path = asr_translate_path_db
            video.translate_subtitle = subtitle_translate_path_db
            vtt_path = srt2vtt(curPath + '/' + subtitle_translate_path_db)
            video.translate_subtitle = vtt_path
            video.save()
            print("translate end")

            file_name = curPath + '/' + video.translate_subtitle
            time_pair = []
            with open(file_name, 'r') as f:
                line = f.readline()
                while line:
                    if '-->' in line:
                        line = line.strip().split('-->')
                        line[0] = line[0].strip()
                        line[1] = line[1].strip()
                        start_time = line[0][0:-4]
                        stop_time = line[1][0:-4]

                        start_time = start_time.split(':')
                        start_time = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + int(start_time[2]) * 1

                        stop_time = stop_time.split(':')
                        stop_time = int(stop_time[0]) * 3600 + int(stop_time[1]) * 60 + int(stop_time[2]) * 1

                        time_pair.append([start_time, stop_time])
                        # print(line)
                    line = f.readline()
            print("time_pair", time_pair)
            index = 0
            while index < len(time_pair) - 1:
                print(time_pair[index][1], time_pair[index + 1][0])
                if time_pair[index + 1][0] - time_pair[index][1] < 3:
                    time_pair[index + 1][0] = time_pair[index][0]
                    time_pair.remove(time_pair[index])
                    index -= 1
                index += 1
            print("time_pair", time_pair)
            feature_list = []
            people_all = People.objects.all()
            for people in people_all:
                feature_item = []
                feature_item.append(people.name)
                feature_item.append(people.voice_feature_path)
                feature_list.append(feature_item)
                print(feature_item)
            print(feature_list)

            video_path = curPath + '/' + video.rel_path
            ans = recognition_video_api(feature_list, video_path, time_pair)
            print("ans", ans)
            voiceprint_ans = 'statics/resource/voice/audio_json/' + video.name.split('.')[0] + '.json'
            with open(curPath + '/' + voiceprint_ans, 'w') as load_f:
                json.dump(ans, load_f)
            print('voiceprint_ans', voiceprint_ans)
            video.voice_json = voiceprint_ans
            video.save()
            print("voiceprint done")




    # file_name = '/home/wh/zzw/demo_django/statics/resource/audio_text/Adam_H_Sterling.srt'
    # time_pair = []
    # with open(file_name, 'r') as f:
    #     line = f.readline()
    #     while line:
    #         if '-->' in line:
    #             line = line.strip().split('-->')
    #             line[0] = line[0].strip()
    #             line[1] = line[1].strip()
    #             start_time = line[0][0:-4]
    #             stop_time = line[1][0:-4]
    #
    #             start_time = start_time.split(':')
    #             start_time = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + int(start_time[2]) * 1
    #
    #             stop_time = stop_time.split(':')
    #             stop_time = int(stop_time[0]) * 3600 + int(stop_time[1]) * 60 + int(stop_time[2]) * 1
    #
    #             time_pair.append([start_time, stop_time])
    #             # print(line)
    #         line = f.readline()
    # print("time_pair", time_pair)
    # index = 0
    # while index < len(time_pair) - 1:
    #     print(time_pair[index][1], time_pair[index+1][0])
    #     if time_pair[index+1][0] - time_pair[index][1] < 3:
    #         time_pair[index+1][0] = time_pair[index][0]
    #         time_pair.remove(time_pair[index])
    #         index -= 1
    #     index += 1
    #
    # print("time_pair", time_pair)
    #
    #
    # feature_list = []
    # people_all = People.objects.all()
    # for people in people_all:
    #     feature_item = []
    #     feature_item.append(people.name)
    #     feature_item.append(people.voice_feature_path)
    #     feature_list.append(feature_item)
    #     print(feature_item)
    # print(feature_list)
    #
    # video_path = '/home/wh/zzw/demo_django/statics/resource/videos/Adam_H_Sterling.mp4'
    # ans = recognition_video_api(feature_list, video_path, time_pair)
    # print("ans", ans)
    # with open(curPath + '/statics/resource/voice/audio_json/Adam_H_Sterling.json', 'w') as load_f:
    #     json.dump(ans, load_f)

    return JsonResponse({
        "code": 0,
        "msg": "",
        "status": 1,
        "fileid": 0,
    })

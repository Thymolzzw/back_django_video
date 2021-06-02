import argparse
import os
import time
import uuid
import tensorflow
import numpy as np
from flask import request, Flask, render_template
from flask_cors import CORS

import pymysql
from pydub import AudioSegment

from demo_django.voice import model, utils



#声纹标注的
# 截取音频  文件地址   开始时间   结束时间(00:00)
def Extract_vioce(file_path,start_time,stop_time,voice_name):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)
    snd = AudioSegment.from_file(file_path)
    filename = str(uuid.uuid1()) + '.wav'
    export_path= os.path.join(curPath, 'statics', 'resource', 'voice', 'audio_temp', filename)
    # export_path = 'audio_temp\\test.wav'
    snd.export(export_path, format='wav')  # 导出为wav格式的音频在根目录下
    print("time:", start_time, "~", stop_time)  # 转换为ms
    if len(start_time.split(':'))==3:
        start_time = (int(start_time.split(':')[0]) * 3600+int(start_time.split(':')[1]) * 60 + int(start_time.split(':')[2])) * 1000
    else:
        start_time = (int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])) * 1000

    if len(stop_time.split(':')) == 3:
        stop_time = (int(stop_time.split(':')[0]) * 3600 + int(stop_time.split(':')[1]) * 60 + int(stop_time.split(':')[2])) * 1000
    else:
        stop_time = (int(stop_time.split(':')[0]) * 60 + int(stop_time.split(':')[1])) * 1000
    print("ms:", start_time, "~", stop_time)
    sound = AudioSegment.from_mp3(export_path)
    if start_time<0 or stop_time>len(sound):#检测时间问题
        print("时间超出范围")
        return 0
    else:
        crop_audio = sound[start_time:stop_time]  # 进行截取
        save_name = voice_name + '.wav'  # 截取的声音保存在audio_db下
        save_name = os.path.join(curPath, 'statics', 'resource', 'voice', 'audio_db', save_name)
        crop_audio.export(save_name, format="wav")
        print("export:" + export_path)
        os.remove(export_path)
        return save_name

#  声纹识别的音频提取
def Extract_vioce_recognition(file_path,start_time,stop_time):
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)
    print("file_path", file_path)
    snd = AudioSegment.from_file(file_path)
    print("snd", snd)
    filename = str(uuid.uuid1()) + '.wav'
    export_path = os.path.join(curPath, 'statics', 'resource', 'voice', 'audio_temp', filename)
    # export_path = 'audio_temp\\test.wav'

    snd.export(export_path, format='wav')  # 导出为wav格式的音频在根目录下
    print("time:", start_time, "~", stop_time)  # 转换为ms

    start_time = start_time * 1000

    stop_time = stop_time * 1000

    print("ms:", start_time, "~", stop_time)
    sound = AudioSegment.from_mp3(export_path)
    os.remove(export_path)
    if start_time < 0:
        start_time = 0
    if stop_time > len(sound):
        stop_time = len(sound)

    crop_audio = sound[start_time:stop_time]  # 进行截取
    print('crop_audio', crop_audio)
    save_name = str(uuid.uuid1()) + '.wav'  # 截取的声音保存在audio_db下
    save_name = os.path.join(curPath, 'statics', 'resource', 'voice', 'audio_temp', save_name)
    crop_audio.export(save_name, format="wav")
    return save_name

# 获取声纹特征
def predict(audio_path,voice_name,params, network_eval):
    specs = utils.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                            hop_length=params['hop_length'], n_fft=params['nfft'],
                            spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    feature = network_eval.predict(specs)[0]
    save_path= voice_name+'.npy'  # 保存权重 权重名称就是声纹名称
    save_path=os.path.join(curPath, 'statics', 'resource', 'voice', 'audio_feature', save_path)
    #print(save_path)
    np.save(save_path,feature)
    #print(feature)
    return feature,save_path

#保存权重
def f_out(file_path):
    f=np.load(file_path)
    print(f)

#声纹标注器   视频文件地址   开始时间  结束时间   声纹名称
def vedio_to_viocefeature(file_path,start_time,stop_time,voice_name, params, network_eval):
    if not os.path.exists(file_path):#防止文件不存在
        print("wrong file path!")
        return 0
    else:
        voice_path = Extract_vioce(file_path, start_time, stop_time, voice_name)
        if voice_path==0:
            return 0
        else:
            save_path = predict(voice_path, voice_name, params, network_eval)
            #print(os.path.dirname(os.path.abspath(save_path[1])))
            sql_save(os.path.join('statics', 'resource', 'voice', 'audio_feature', save_path[1].split(os.sep)[-1]), voice_name)

            return 1

#将声纹特征存入数据库中
def sql_save(file_path,feature_name):
    # 连接数据库
    count = pymysql.connect(
        host='127.0.0.1',  # 数据库地址
        port=3306,  # 数据库端口号
        user='root',  # 数据库账号
        password='root',  # 数据库密码
        db='test_db')  # 数据库名称
    # 创建数据库对象
    db = count.cursor()
    # 写入SQL语句
    sql = "update people set voice_feature_path=%s where name = %s;"
    print(sql)
    try:
        # 执行sql
        db.execute(sql, [file_path,feature_name])
        # 提交事务
        count.commit()
        print('插入成功')
    except Exception as e:
        print(e)
        count.rollback()
        print('插入失败')
    finally:
        count.close()
    db.close()


#读取数据库中的声纹特征
def sql_find():
    # 连接数据库
    count = pymysql.connect(
        host='127.0.0.1',  # 数据库地址
        port=3306,  # 数据库端口号
        user='root',  # 数据库账号
        password='root',  # 数据库密码
        db='test_db')  # 数据库名称
    # 创建数据库对象
    db = count.cursor()
    # 写入SQL语句
    sql = "select * from people where voice_feature_path != 'NULL'"
    try:
        # 执行sql
        db.execute(sql)
        # 提交事务
        result = db.fetchall()
        db.close()
        return result#返回结果
    except Exception as e:
        print(e)
        count.rollback()
        print('加载特征失败')
    finally:
        count.close()
    db.close()

#声纹识别中获取要预测的声纹特征     注意！与predict是不同的，predict_voice不保存音频，只返回音频特征
def predict_voice(audio_path, params, network_eval):
    print("enter predict_voice")
    specs = utils.load_data(audio_path, win_length=params['win_length'], sr=params['sampling_rate'],
                            hop_length=params['hop_length'], n_fft=params['nfft'],
                            spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    feature = network_eval.predict(specs)[0]
    print("out predict_voice")
    return feature


#输入视频地址  视频中的开始时间  结束时间
def recognition_video(feature_list,file_path,start_time,stop_time, params, network_eval):
    print("Extract_vioce_recognition in")
    voice_path = Extract_vioce_recognition(file_path, start_time, stop_time)  # 先从视频提取对应部分的音频
    print("Extract_vioce_recognition out")
    return_ans = zzw_recognition_voice(feature_list,voice_path, params, network_eval)
    os.remove(voice_path)
    return return_ans

def zzw_recognition_voice(feature_list,file_path, params, network_eval):
    print("file_path", file_path)

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)
    name = ''
    pro = 0
    try:
        feature = predict_voice(file_path, params, network_eval)#提取要识别的音频特征
        # print('feature', feature)
        for i in feature_list:  # i[1]是特征文件地址  i[0]是声纹特征标注的名称
            if i[1] != None:
                dist = np.dot(feature, np.load(os.path.join(curPath, i[1])).T)
                if dist > pro:
                    pro = dist
                    name = i[0]
        result = {}
        if name != '':
            result['score'] = pro
            result['name'] = name
        else:
            result = None
        return result   #result格式：[0.7687116, 'wj']  result[0]是概率   result[1]是名字
    except Exception as e:
        print(e)
        pass
    return None


def register_voice_api():
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)

    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True}

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval')  # args=args

    # ==> load pre-trained model
    network_eval.load_weights(curPath + '/' + 'demo_django/voice/' + 'pretrained/weights.h5', by_name=True)

    print("进入")
    file_path = '/home/wh/zzw/demo_django/demo_django/voice/newpeoplevoice/Kobe.mp4'
    start_time = '0:05'
    stop_time = '1:05'
    voice_name = 'Kobe'

    vedio_to_viocefeature(file_path, start_time, stop_time, voice_name, params, network_eval)


def recognition_video_api(feature_list, file_path, time_list):
    print("time_list", len(time_list))

    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)

    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True}

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval')  # args=args
    # ==> load pre-trained model
    network_eval.load_weights(curPath + '/' + 'demo_django/voice/' + 'pretrained/weights.h5', by_name=True)
    i = 0
    return_ans = []
    for t in time_list:
        print("i", i)

        item = {}
        start = int(t[0])
        stop = int(t[1])
        print('start', start, stop)
        if stop - start > 100:
            stop = start + 100
        print(start, stop)
        ans = recognition_video(feature_list, file_path, start, stop, params, network_eval)
        print("ans", ans)
        if ans != None:
            item['score'] = int(ans['score'].item() * 1000) / 1000
            item['name'] = ans['name']
            start = formate_time(int(t[0]))
            stop = formate_time(int(t[1]))
            item['time'] = str(start) + ' --> ' + str(stop)
            return_ans.append(item)
        i += 1
    if len(return_ans) > 0:
        return return_ans
    else:
        return None

def formate_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

#将特征文件存为npy结尾的文件
if __name__ == '__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    split_reg = 'demo_django'
    curPath = curPath.split(split_reg)[0] + split_reg
    print(curPath)

    person_feature = []
    person_name = []

    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tensorflow.compat.v1.Session(config=config)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True}

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval')  # args=args

    # ==> load pre-trained model
    network_eval.load_weights(curPath + '/' + 'demo_django/voice/' + 'pretrained/weights.h5', by_name=True)

    register_voice_api()

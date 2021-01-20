import json
import os
import socket

import pymysql
# 连接数据库
# count = pymysql.connect(
#             host='localhost',   # 数据库地址
#             port=3306,    # 数据库端口号
#             user='root',    # 数据库账号
#             password='root',    # 数据库密码
#             db='test_db')    # 数据库名称
# # 创建数据库对象
# db = count.cursor()
# # 写入SQL语句
# sql = "select * from users "
# # 执行sql命令
# db.execute(sql)
# # 获取一个查询
# # restul = db.fetchone()
# # 获取全部的查询内容
# restul = db.fetchall()
# print(restul)
# db.close()

# import time
#
# t = time.time()
# s = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
# print(int(t))
# print(s)
#
#
# import numpy as np
# curPath = os.path.abspath(os.path.dirname(__file__))
# split_reg = curPath.split(os.sep)[-1]
# curPath = curPath.split(split_reg)[0] + split_reg
# print("111" + curPath)
# loadData = np.load(os.path.join("D:\TensorFlow_workplace_new\demo_django\statics\\resource\\text", 'res_face.npy'), allow_pickle=True)
# print(loadData)
# data = []
# for item in loadData:
#     print(item)
#     item_item = {"name": item[0]}
#     item_list = []
#     for i in range(len(item[1])):
#         item_list.append({
#             'time': item[1][i],
#             'img': item[2][i]
#         })
#     print(item_list)
#     item_item["time_img"] = item_list
#     data.append(item_item)
#
# res = {
#         "code": 0,
#         "msg": "",
#         "status": 1,
#         "data": data
#     }
# print(res)


# import speech_recognition as sr
#
# # obtain audio from the microphone
# r = sr.Recognizer()
# harvard = sr.AudioFile(r"D:\TensorFlow_workplace_new\wj\Kersa-Speaker-Recognition-master\audio\tingli2.wav")
# with harvard as source:
#     audio = r.record(source)
# # recognize speech using Sphinx
# try:
#     print("Sphinx thinks you said \n " + r.recognize_sphinx(audio))
# except sr.UnknownValueError:
#     print("Sphinx could not understand audio")
# except sr.RequestError as e:
#     print("Sphinx error; {0}".format(e))

#
# import uuid
#
#
# print(type(str(uuid.uuid1())))

s = socket.create_connection(("www.google.com", 80), 2)
s.close()


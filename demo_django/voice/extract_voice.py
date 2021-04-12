import pathlib
from pydub import AudioSegment
import os

# 提取音频
# 截取音频  文件地址   开始时间   结束时间(00:00)
def Extract_vioce(file_path,start_time,stop_time,voice_name):
    snd = AudioSegment.from_file(file_path)
    export_path = 'test.wav'
    snd.export(export_path,format='wav')  # 导出为wav格式的音频在根目录下
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
        save_name = "audio_db\\" + voice_name + '.wav'  # 截取的声音保存在audio_db下
        crop_audio.export(save_name, format="wav")
        os.remove(export_path)
        return save_name




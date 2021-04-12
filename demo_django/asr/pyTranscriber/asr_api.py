from demo_django.asr.pyTranscriber.pytranscriber.control.new_ctr_main import Ctr_Main


def asr_subtitle(file, output):
    ctrMain = Ctr_Main()
    ctrMain.listenerBExec(file, output)
    print('done')

if __name__ == '__main__':
    file = '/home/wh/zzw/demo_django/statics/resource/videos/zzw.mp4'
    output = '/home/wh/下载/'
    asr_subtitle(file, output)


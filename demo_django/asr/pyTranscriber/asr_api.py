from demo_django.asr.pyTranscriber.pytranscriber.control.new_ctr_main import Ctr_Main


def asr_subtitle(file, output):
    ctrMain = Ctr_Main()
    ctrMain.listenerBExec(file, output)
    print('done')

if __name__ == '__main__':
    file = 'D:\download\\video\\test.mp4'
    output = 'D:\download\\video\\'
    asr_subtitle(file, output)


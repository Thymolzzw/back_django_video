import demo_django.face_recignition.cky_remove_same_frame as fr

if __name__ == '__main__':
    file = 'test55.mp4'
    output_npy = 'final_result/FaceRecognition_' + str(file.split('.')[0]) + '_result.npy'
    fr.main(file, output_npy)
    print('done')


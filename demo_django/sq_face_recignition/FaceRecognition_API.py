import os

import demo_django.sq_face_recignition.cky_remove_same_frame as fr
import time
import uuid
import demo_django.sq_face_recignition.train as tr

def face_recognition(file):
    this_module_start = time.time()
    uu = uuid.uuid1()
    curPath = os.path.abspath(os.path.dirname(__file__))
    curPath = curPath.split("demo_django")[0] + "demo_django"

    output_npy_db = 'statics/resource/face_npy/' + str(file.split('/')[-1].split('.')[0]) + '_' + str(uu) + '.npy'
    output_npy = curPath + '/' + output_npy_db
    print("output_npy", output_npy)
    fr.main(file, output_npy)
    print('done')
    this_module_end = time.time()
    print("This module cost " + str(this_module_end - this_module_start) + "s")
    return output_npy_db


if __name__ == '__main__':


    train_file = 'train'

    # curPath = os.path.abspath(os.path.dirname(__file__))
    # curPath = curPath.split("demo_django")[0] + "demo_django"


    print("Training KNN classifier...")
    classifier = tr.train(train_file, model_save_path="model/trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

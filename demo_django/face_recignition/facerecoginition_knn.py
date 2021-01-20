

from imutils import paths
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw,ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image,number_of_times_to_upsample=0, model="cnn")

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(frame, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))
    #
    # if knn_clf is None and model_path is None:
    #     raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = frame
    X_face_locations = face_recognition.face_locations(X_img,number_of_times_to_upsample=0, model="cnn")

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image =Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("utf-8")

        #namefont=ImageFont.truetype("simsun.ttc",20)
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        # draw.rectangle(((left, bottom - text_height - 20), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        # #draw.text((left + 2, bottom - text_height - 10), name, fill=(255, 255, 255, 255),font=namefont)
        # draw.text((left + 2, bottom - text_height -2), name, fill=(255, 255, 255, 255))
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    #pil_image.show()
    return pil_image


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train("datanew/train", model_save_path="model/trained_knn_model.clf", n_neighbors=2)
    # print("Training complete!")

    # Open the input movie file
    input_movie = cv2.VideoCapture('test3.mp4')
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create an output movie file (make sure resolution/frame rate matches input video!)
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    output_movie = cv2.VideoWriter('testoutput3.avi', fourcc, 29.97, (1280, 720))

    frame_number = 0
    list_knownface=[]
    #face_img_save_path = 'Unknown_face/face_imgs/'

    fps = input_movie.get(cv2.CAP_PROP_FPS)
    fps=math.ceil(fps)
    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break
        if(frame_number%fps==0):#1s抽一帧
            rgb_frame = frame[:, :, ::-1]
            predictions = predict(rgb_frame, model_path="model/trained_knn_model.clf")
            # STEP 2: Using the trained classifier, make predictions for unknown images
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance

            i=0
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                if name=='unknown':
                    #UnKnownfacce=Image.fromarray(UnKnownface)
                    #UnKnownfacce.show()
                    cv2.imwrite('Unknown_face/'+str(i)+'.jpg',frame[top:bottom, left:right])
                    i=i+1
                else:
                   list_knownface.append((name,frame_number,frame))
                   #print(list_knownface)

            if predictions:

                #print("11111")
                # Display results overlaid on an image
                imagefinal=show_prediction_labels_on_image(frame, predictions)
                print("Writing frame {} / {}".format(frame_number, length))
                output_movie.write(np.array(imagefinal))

            else:
                print("Writing frame {} / {}".format(frame_number, length))
                output_movie.write(frame)
    #print(list_knownface)

    list_set = set()
    for name, t ,id in list_knownface:
        list_set.add(name)
        # print(list_set)
    # 统计已知人脸姓名关联百度百科
    #for every in iter(list_set):
        #baidu.query(every)
    #已知人脸保存时间戳及对应帧
    res_list = []
    for item in list_set:
        li_time = []
        li_frame=[]
        for list_item in list_knownface:
            if list_item[0] == item:
                li_time.append(list_item[1])
                li_frame.append(list_item[2])
        k = 1
        while k < len(li_time):
            if li_time[k] - li_time[k - 1] <= fps:
                li_time.remove(li_time[k])
                del li_frame[k]
                k -= 1
            k += 1
        list_name=[]
        cnt=0
        for i in li_time:
            img_name = 'time_frame/' + item + str(i) + '.jpg'
            list_name.append(item + str(i) + '.jpg')
            img_item = cv2.imwrite(img_name, li_frame[cnt])
            cnt+=1

        res_list.append((item, li_time, list_name))

    np.save('res_list.npy',np.array(res_list))
    datares=np.load('res_list.npy',allow_pickle=True)
    print(datares)


    # All done!
    input_movie.release()
    cv2.destroyAllWindows()

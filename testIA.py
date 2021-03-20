from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
from codeReconnaissance import detect_face, facenet
import os
import time
import pickle
import sys
from PIL import Image
import glob
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.codeBdd import interroBdd
from src.codeBdd import Client
import matplotlib.pyplot as plt

M = interroBdd.Interro('Clients', 'Clients')
M.creation()
M.ajoutColonne('Nom', 'TEXT', 'nomClient')
M.ajoutColonne('Prenom', 'TEXT', 'prenomClient')
M.ajoutColonne('Path', 'TEXT', 'Oui')
M.ajoutColonne('Dette', 'INTEGER', 0.0)
M.ajoutColonne('tauxAlcool', 'INTEGER', 0.0)

print("la bdd a été créée")  # test

i = 1
list = glob.glob('../train_img/*')
for x in list:
    client = Client(x)
    M.insertion(i, 'Nom', 0.0)
    M.update(i, 'Nom', client.nom)
    M.update(i, 'Prenom', client.prenom)
    M.update(i, 'Path', x)
    M.update(i, 'Dette', int(client.dette))
    M.update(i, 'tauxAlcool', int(client.tauxAlcool))
    i += 1

print("la bdd a été mise  jour")  # test

# tests des fcts
M.update(1, 'Dette', 45)
M.update(2, 'Dette', 74)
M.update(3, 'Dette', 99)
M.update(4, 'Dette', 50000)
M.update(5, 'Dette', 0.5)

M.update(1, 'tauxAlcool', 0.2)
M.update(2, 'tauxAlcool', 0.55)
M.update(3, 'tauxAlcool', 3)
M.update(4, 'tauxAlcool', 89)
M.update(5, 'tauxAlcool', 1.6)


# img_path = '../ressources/abc.jpg'







modeldir = '../model/20170511-185253.pb'
classifier_filename = '../class/classifier.pkl'
npy = '../npy'
train_img = "../train_img"


def identify_face_img(img_path):
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading feature extraction model')
            facenet.load_model(modeldir)

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)


            c = 0

            print('Start Recognition!')
            prevTime = 0


            frame = cv2.imread(img_path, 0)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

            curTime = time.time() + 1  # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Face Detected: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(
                            np.array(Image.fromarray(cropped[i]).resize((image_size, image_size), Image.BILINEAR)))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print(best_class_probabilities)
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                        # plot result idx under box
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        nom_prenom = HumanNames[best_class_indices[0]]
                        # img_pred.append(nom_prenom)
                        nom = nom_prenom.split()[0]
                        prenom = nom_prenom.split()[1]


                        for H_i in HumanNames:
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                        return nom_prenom
                else:
                    print('Unable to align / Unknown')
                    nom_prenom = 'Unknown'
            # cv2.imshow('Image', frame)
            return nom_prenom

            if cv2.waitKey(1000000) & 0xFF == ord('q'):
                sys.exit("Thanks")
            cv2.destroyAllWindows()


def obstacle_rect_hg(imgName): #obstruction d'une image avec un rectangle dans le quart supérieur gauche
    img = cv2.imread(imgName)
    L, l, _ = img.shape #largeur, longueur
    # print("largeur/longueur/profondeur :",l, L, _)
    NewImg = cv2.rectangle(img, (0, 0), (int(l/2), int(L/2)), (0, 0, 0), -1)
    nom = imgName + "obstacleRectHG" + ".png"
    cv2.imwrite(nom, NewImg)

def obstacle_rect_bg(imgName):  # obstruction d'une image avec un rectangle dans le quart inférieur gauche
    img = cv2.imread(imgName)
    L, l, _ = img.shape #largeur, longueur
    # print("largeur/longueur/profondeur :",l, L, _)
    NewImg = cv2.rectangle(img, (0, int(L/2)), (int(l/2), L), (0, 0, 0), -1)
    nom = imgName + "obstacleRectBG" + ".png"
    cv2.imwrite(nom, NewImg)

def obstacle_rect_hd(imgName):  # obstruction d'une image avec un rectangle dans le quart supérieur droit
    img = cv2.imread(imgName)
    L, l, _ = img.shape #largeur, longueur
    # print("largeur/longueur/profondeur :",l, L, _)
    NewImg = cv2.rectangle(img, (int(l/2), 0), (l, int(L/2)), (0, 0, 0), -1)
    nom = imgName + "obstacleRectHD" + ".png"
    cv2.imwrite(nom, NewImg)

def obstacle_rect_bd(imgName):  # obstruction d'une image avec un rectangle dans le quart inférieur droit
    img = cv2.imread(imgName)
    L, l, _ = img.shape #largeur, longueur
    # print("largeur/longueur/profondeur :",l, L, _)
    NewImg = cv2.rectangle(img, (int(l/2), int(L/2)), (l, L), (0, 0, 0), -1)
    nom = imgName + "obstacleRectBD" + ".png"
    cv2.imwrite(nom, NewImg)

def floutage(imgName):
    img = cv2.imread(imgName)
    NewImg = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    # NewImg = cv2.GaussianBlur(img,(11,11),cv2.BORDER_DEFAULT)
    nom = imgName + "Floutee" + ".png"
    cv2.imwrite(nom, NewImg)

if __name__ == "__main__":

    imgs_test_normal = ['../ressources/img_test1.jpeg',  # Akshay Kumar
                 '../ressources/img_test2.jpeg',  # Shahrukh Khan
                 '../ressources/img_test3.jpeg',  # Nawazuddin Siddiqui
                 '../ressources/img_test4.jpeg',  # Sunil Shetty
                 '../ressources/img_test5.jpeg',  # Akshay Kumar
                 '../ressources/img_test6.jpeg',  # Sunil Shetty
                 '../ressources/img_test7.jpeg',  # Nawazuddin Siddiqui
                 '../ressources/img_test8.jpeg',  # Sunny Deol
                 '../ressources/img_test9.jpeg',  # Salman Khan
                 '../ressources/img_test10.jpeg',  # Sunny Deol
                 '../ressources/img_test11.jpeg',  # Shahrukh Khan
                 '../ressources/img_test12.jpeg']  # Salman Khan




    img_target_normal = ['Akshay Kumar',
                  'Shahrukh Khan',
                  'Nawazuddin Siddiqui',
                  'Sunil Shetty',
                  'Akshay Kumar',
                  'Sunil Shetty',
                  'Nawazuddin Siddiqui',
                  'Sunny Deol',
                  'Salman Khan',
                  'Sunny Deol',
                  'Shahrukh Khan',
                  'Salman Khan']

    target_names = ['Akshay Kumar',
                    'Shahrukh Khan',
                    'Sunny Deol',
                    'Salman Khan',
                    'Sunil Shetty',
                    'Nawazuddin Siddiqui'] #,
                    #'Unknown']

    img_pred_normal = []

    for img_path in imgs_test_normal:
        img_pred_normal.append(identify_face_img(img_path))

    print("********************** Predicting people's names on the test set normal ******************************")

    print("Rapport de classification :")
    print(classification_report(img_target_normal, img_pred_normal, target_names=target_names))
    print("Matrice de confusion :")
    print(confusion_matrix(img_target_normal, img_pred_normal, labels=target_names))




################################ test avec des obstacles qui obstruent les visages ###################################"""
    # Ici on crée les images avec des obstacles
    for imgName in imgs_test_normal:
        obstacle_rect_hg(imgName)
        obstacle_rect_bg(imgName)
        obstacle_rect_hd(imgName)
        obstacle_rect_bd(imgName)

    imgs_test_obstacle = ['../ressources/img_test1.jpegobstacleRectBD.png', '../ressources/img_test1.jpegobstacleRectBG.png','../ressources/img_test1.jpegobstacleRectHD.png','../ressources/img_test1.jpegobstacleRectHG.png', # Akshay Kumar
                 '../ressources/img_test2.jpegobstacleRectBD.png', '../ressources/img_test2.jpegobstacleRectBG.png','../ressources/img_test2.jpegobstacleRectHD.png','../ressources/img_test2.jpegobstacleRectHG.png', # Shahrukh Khan
                 '../ressources/img_test3.jpegobstacleRectBD.png', '../ressources/img_test3.jpegobstacleRectBG.png','../ressources/img_test3.jpegobstacleRectHD.png','../ressources/img_test3.jpegobstacleRectHG.png',  # Nawazuddin Siddiqui
                 '../ressources/img_test4.jpegobstacleRectBD.png', '../ressources/img_test4.jpegobstacleRectBG.png','../ressources/img_test4.jpegobstacleRectHD.png','../ressources/img_test4.jpegobstacleRectHG.png',  # Sunil Shetty
                 '../ressources/img_test5.jpegobstacleRectBD.png', '../ressources/img_test5.jpegobstacleRectBG.png','../ressources/img_test5.jpegobstacleRectHD.png','../ressources/img_test5.jpegobstacleRectHG.png',  # Akshay Kumar
                 '../ressources/img_test6.jpegobstacleRectBD.png', '../ressources/img_test6.jpegobstacleRectBG.png','../ressources/img_test6.jpegobstacleRectHD.png','../ressources/img_test6.jpegobstacleRectHG.png',  # Sunil Shetty
                 '../ressources/img_test7.jpegobstacleRectBD.png', '../ressources/img_test7.jpegobstacleRectBG.png','../ressources/img_test7.jpegobstacleRectHD.png','../ressources/img_test7.jpegobstacleRectHG.png',  # Nawazuddin Siddiqui
                 '../ressources/img_test8.jpegobstacleRectBD.png', '../ressources/img_test8.jpegobstacleRectBG.png','../ressources/img_test8.jpegobstacleRectHD.png','../ressources/img_test8.jpegobstacleRectHG.png',  # Sunny Deol
                 '../ressources/img_test9.jpegobstacleRectBD.png', '../ressources/img_test9.jpegobstacleRectBG.png','../ressources/img_test9.jpegobstacleRectHD.png','../ressources/img_test9.jpegobstacleRectHG.png',  # Salman Khan
                 '../ressources/img_test10.jpegobstacleRectBD.png', '../ressources/img_test10.jpegobstacleRectBG.png','../ressources/img_test10.jpegobstacleRectHD.png','../ressources/img_test10.jpegobstacleRectHG.png',  # Sunny Deol
                 '../ressources/img_test11.jpegobstacleRectBD.png', '../ressources/img_test11.jpegobstacleRectBG.png','../ressources/img_test11.jpegobstacleRectHD.png','../ressources/img_test11.jpegobstacleRectHG.png',  # Shahrukh Khan
                 '../ressources/img_test12.jpegobstacleRectBD.png', '../ressources/img_test12.jpegobstacleRectBG.png','../ressources/img_test12.jpegobstacleRectHD.png','../ressources/img_test12.jpegobstacleRectHG.png']  # Salman Khan

    img_target_obstacle = ['Akshay Kumar','Akshay Kumar','Akshay Kumar','Akshay Kumar',
                            'Shahrukh Khan','Shahrukh Khan','Shahrukh Khan','Shahrukh Khan',
                            'Nawazuddin Siddiqui','Nawazuddin Siddiqui','Nawazuddin Siddiqui','Nawazuddin Siddiqui',
                             'Sunil Shetty','Sunil Shetty','Sunil Shetty','Sunil Shetty',
                             'Akshay Kumar','Akshay Kumar','Akshay Kumar','Akshay Kumar',
                             'Sunil Shetty','Sunil Shetty','Sunil Shetty','Sunil Shetty',
                              'Nawazuddin Siddiqui','Nawazuddin Siddiqui','Nawazuddin Siddiqui','Nawazuddin Siddiqui',
                               'Sunny Deol','Sunny Deol','Sunny Deol','Sunny Deol',
                              'Salman Khan','Salman Khan','Salman Khan','Salman Khan',
                              'Sunny Deol','Sunny Deol','Sunny Deol','Sunny Deol',
                             'Shahrukh Khan','Shahrukh Khan','Shahrukh Khan','Shahrukh Khan',
                             'Salman Khan','Salman Khan','Salman Khan','Salman Khan']

    target_names_obstcales = ['Akshay Kumar',
                    'Shahrukh Khan',
                    'Sunny Deol',
                    'Salman Khan',
                    'Sunil Shetty',
                    'Nawazuddin Siddiqui',
                    'Unknown']
    img_pred_obstacle = []
    #
    for img_path in imgs_test_obstacle:
        img_pred_obstacle.append(identify_face_img(img_path))

    print("********************** Predicting people's names on the test set with obstacles ******************************")

    print("Rapport de classification (avec obstacles) :")
    print(classification_report(img_target_obstacle, img_pred_obstacle, target_names=target_names_obstcales))
    print("Matrice de confusion (avec obstacles) :")
    print(confusion_matrix(img_target_obstacle, img_pred_obstacle, labels=target_names_obstcales))


###################### test sur des visages floutés ###################################"""
    # Ici on crée les images floutées
    for imgName in imgs_test_normal:
        floutage(imgName)

    imgs_test_floue = ['../ressources/img_test1.jpegFloutee.png',  # Akshay Kumar
                 '../ressources/img_test2.jpegFloutee.png',  # Shahrukh Khan
                 '../ressources/img_test3.jpegFloutee.png',  # Nawazuddin Siddiqui
                 '../ressources/img_test4.jpegFloutee.png',  # Sunil Shetty
                 '../ressources/img_test5.jpegFloutee.png',  # Akshay Kumar
                 '../ressources/img_test6.jpegFloutee.png',  # Sunil Shetty
                 '../ressources/img_test7.jpegFloutee.png',  # Nawazuddin Siddiqui
                 '../ressources/img_test8.jpegFloutee.png',  # Sunny Deol
                 '../ressources/img_test9.jpegFloutee.png',  # Salman Khan
                 '../ressources/img_test10.jpegFloutee.png',  # Sunny Deol
                 '../ressources/img_test11.jpegFloutee.png',  # Shahrukh Khan
                 '../ressources/img_test12.jpegFloutee.png']  # Salman Khan




    img_target_floue = ['Akshay Kumar',
                  'Shahrukh Khan',
                  'Nawazuddin Siddiqui',
                  'Sunil Shetty',
                  'Akshay Kumar',
                  'Sunil Shetty',
                  'Nawazuddin Siddiqui',
                  'Sunny Deol',
                  'Salman Khan',
                  'Sunny Deol',
                  'Shahrukh Khan',
                  'Salman Khan']

    target_names_floue = ['Akshay Kumar',
                    'Shahrukh Khan',
                    'Sunny Deol',
                    'Salman Khan',
                    'Sunil Shetty',
                    'Nawazuddin Siddiqui',
                    'Unknown']

    img_pred_floue = []

    for img_path in imgs_test_floue:
        img_pred_floue.append(identify_face_img(img_path))

    print("********************** Predicting people's names on the test set normal ******************************")

    print("Rapport de classification (floue) :")
    print(classification_report(img_target_floue, img_pred_floue, target_names=target_names_floue))
    print("Matrice de confusion (floue) :")
    print(confusion_matrix(img_target_floue, img_pred_floue, labels=target_names_floue))
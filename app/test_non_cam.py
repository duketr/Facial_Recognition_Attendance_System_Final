# coding=utf-8

# Author
#
# Dang BH
#
# KSE Company

import sys
sys.path.append('../')
sys.path.append('gui_app/')
import configparser

conf = configparser.ConfigParser()
conf.read("../config/main.cfg")

# import some PyQt5 modules
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QAction
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QMovie
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer,QThread,pyqtSignal,QObject

from gui_app.ui_2 import *
# from login import *
# from chupCam import *
from gui_app.noti_gui import *

import tensorflow as tf
import imutils
import re
import os
import sklearn
import onnxruntime
from utils import face_preprocess
from Detection.mask_detector.face_detector import FaceDetector
from Detection.mask_detector.config import *
from utils.ROI_extraction import extract_ROI
from utils.api import send_data_to_database

from datetime import datetime as dt
import re

import time
# import connect_DB

import loadModels

import numpy as np
import cv2

MODEL_PATH = conf.get("MOBILEFACENET", "MODEL_PATH")
VERIFICATION_THRESHOLD = float(conf.get("MOBILEFACENET", "VERIFICATION_THRESHOLD"))
FACE_DB_PATH = conf.get("MOBILEFACENET", "FACE_DB_PATH")
mtcnn_detector = loadModels.load_mtcnn(conf)
model = onnxruntime.InferenceSession("/Users/duke/Downloads/Facial_Recognition_Attendance_System_Final/Detection/mask_detector/models/mask_detector.onnx", None)
detection_model = FaceDetector("/Users/duke/Downloads/Facial_Recognition_Attendance_System_Final/Detection/mask_detector/models/scrfd_500m.onnx")

def draw_rect(faces, names, sims, image):
    for i, face in enumerate(faces):
        # prob = '%.2f' % sims[i]
        if names[i] == "unknown":
            label = "{}".format(names[i])
        else:
            # name,fullName,position = connect_DB.getEmployee(names[i])
            # label = "{}".format(name)
            print(names[i])
            label = "{}".format(names[i])
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        x, y = int(face[0]), int(face[1])
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def feature_compare(feature1, feature2, threshold):
    dist = np.sum(np.square(feature1- feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim

with tf.Graph().as_default():
    with tf.compat.v1.Session() as sess:
        
        loadModels.load_mobilefacenet(MODEL_PATH)
        inputs_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        print('load model va sess DONE')

        # faces_db = loadModels.load_faces(FACE_DB_PATH, mtcnn_detector,sess,inputs_placeholder,embeddings)
        #--- check if predate has modified -> update
        # connect_DB.checkUpdateImg(status='predate')
        #--- check all ids people in local and server to find anything changed -> update
        # connect_DB.autoUpdateDB()
        # print("update folder DB thanh cong")

        
        class MainWindow(QWidget):
            # class constructor
            accept = pyqtSignal()
            def __init__(self):
                # call QWidget constructor
                super().__init__()
                self.ui = Ui_Form()
                self.ui.setupUi(self)
                
                self.faces_db = loadModels.load_faces(FACE_DB_PATH, mtcnn_detector,sess,inputs_placeholder,embeddings)
                print('load DB Done')

                # self.loginW = LoginWindow()
                # self.camW = TakePhotoWindow()
                # self.loadW = LoadingWin()
                # self.timer_2 = QTimer()

                size = QtWidgets.QDesktopWidget().screenGeometry()
                self.h = size.height()
                self.w = size.width()

                self.attend_arr = []
                self.id_set = set()
                self.frame_detect =0
                        
                # create a timer
                self.timer = QTimer()
                self.start_webcam()
                # self.loginW.openAg.connect(self.outLogin)
            def start_webcam(self):
                self.timer.start(5)
                self.timer.timeout.connect(self.update_frame)
            def face_recog(self,frame):
                faces,landmarks = mtcnn_detector.detect(frame)
                m_faces, _, cropped_face = detection_model.inference(frame)
                flag = ""
                # print('DONE')
                # face_included_frames = 0
                
                if faces.shape[0] is not 0:
                    self.frame_detect+=1
                    # print(self.frame_detect)
                    # if face_included_frames
                    input_images = np.zeros((faces.shape[0], 112,112,3))
                    # print('DONE-2')
                    for i, face in enumerate(faces):
                        if round(faces[i, 4], 6) > 0.95:
                            bbox = faces[i,0:4]
                            points = landmarks[i,:].reshape((5,2))
                            # print(points)
                            nimg = face_preprocess.preprocess(frame, bbox, points, image_size='112,112')
                                

                            # cv2.imshow("face", nimg)
                            nimg = nimg - 127.5
                            nimg = nimg * 0.0078125
                            # input_image = np.expand_dims(nimg, axis=0)
                            input_images[i,:] = nimg
                    # print('DONE-2.5')
                    feed_dict = {inputs_placeholder: input_images}
                    # print('DONE-2.6')
                    emb_arrays = sess.run(embeddings, feed_dict=feed_dict)
                    # print('DONE-2.7')
                    emb_arrays = sklearn.preprocessing.normalize(emb_arrays)
                    # print('DONE-3')
                    # print(emb_arrays)
                    ids = []
                    sims = []
                    if self.frame_detect % 10 == 0:
                        for i, embedding in enumerate(emb_arrays):
                            embedding = embedding.flatten()
                            temp_dict = {}
                            for com_face in self.faces_db:
                                ret, sim = feature_compare(embedding, com_face["feature"], 0.65)
                                temp_dict[com_face["name"]] = sim
                            dict = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
                            print(dict)
                            if dict == []:
                                id = "unknown"
                                sim = 0
                                flag = "Unmasked"
                            else:
                                if dict[0][1] > VERIFICATION_THRESHOLD:
                                    # if '_'in dict[0][0]:
                                    #     id = dict[0][0].split('_')[0]
                                    # else:
                                    #     id = dict[0][0]

                                    id = dict[0][0]
                                    # print(id)

                                    sim = dict[0][1]
                                    flag = "Unmasked"
                                else:
                                    id = "unknown"
                                    sim = 0
                                    flag = "Unmasked"
                            # print('DONE-4')
                            ids.append(id)
                            sims.append(sim)
                            x1, y1, x2, y2 = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                            x1 = max(int(x1), 0)
                            y1 = max(int(y1), 0)
                            x2 = min(int(x2), frame.shape[1])
                            y2 = min(int(y2), frame.shape[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        draw_rect(faces, ids, sims, frame)
                    else:
                        x1, y1, x2, y2 = faces[i][0], faces[i][1], faces[i][2], faces[i][3]
                        x1 = max(int(x1), 0)
                        y1 = max(int(y1), 0)
                        x2 = min(int(x2), frame.shape[1])
                        y2 = min(int(y2), frame.shape[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                elif (len(m_faces) > 0):
                    self.frame_detect += 1
                    ids = []
                    sims = []
                    flag = "Masked"
                    if self.frame_detect % 10 == 0:
                        com_hist = []                        
                        for face in m_faces:
                            face_img = face.cropped_face
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR )
                            face_img = cv2.resize(face_img, (width, height))
                            face_img = face_img.astype(np.float32)
                            face_img = face_img / 255.0
                            face_img = face_img.reshape(1, width, height, 3)
                            y_pred = model.run(['dense_1'], {'conv2d_input' : face_img})
                            prediction = np.argmax(y_pred)
                            hist = extract_ROI(frame, face.bbox)
                            com_hist.append(hist)
                            cv2.rectangle(frame, (int(face.bbox[0]), int(face.bbox[1])), (int(face.bbox[2]), int(face.bbox[3])), (0, 255, 0), 2)
                        
                        for com_face_hist in com_hist:
                            result = []
                            for db_face in self.faces_db:
                                id = db_face['name']
                                sim = np.dot(com_face_hist, db_face['hist'])
                                result.append((id, sim))
                                ids.append(id)
                                sims.append(sim)
                            print("result:", result)

                            max_sim_result = max(result, key=lambda x: x[1])
                            max_id = max_sim_result[0]
                            max_similarity = None
                            for id, sim in result:
                                if id == max_id:
                                    max_similarity = sim
                                    break

                            print("max_id:", max_id)
                            print("max_similarity:", max_similarity)
                            cv2.putText(frame, max_id, (int(face.bbox[0]), int(face.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        return frame, np.array(m_faces), [max_id], [max_similarity], flag
                else:
                    ids = []
                    sims = []
                    flag = "Unmasked"
                return frame,faces,ids,sims,flag
            def update_frame(self):
                cap = cv2.VideoCapture(0)
                ret, frame = cap.read()
                frame,faces,ids,sims,flag = self.face_recog(frame)
                # get image infos
                qImg = self.display_Image(frame)
                if sims == []:
                    pass
                else:
                    for i, face in enumerate(faces):
                        
                        if sims[i] > VERIFICATION_THRESHOLD and flag == "Unmasked":
                            
                            # check = {'id':ids[i],'name':'','fullName':'','position':'','acc':sims[i],'time':dt.now()}
                            check = {'name':ids[i], 'acc':sims[i], 'time':dt.now()}
                            db_name = ids[i]
                            db_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                            send_data_to_database(db_name, db_time)
                            # change for non connect DB
                            self.confW = NotiWindow()    
                            self.timer.stop()
                            self.confW.show()
                            self.confW.setNoti(check,qImg)     

                            # self.confW.autoRecord.connect(lambda: self.accept_db(check,i,faces,landmarks,frame))
                            
                            self.confW.autoStart.connect(self.timer.start)
                        elif flag == "Masked":
                            check = {'name':ids[i], 'acc':sims[i], 'time':dt.now()}
                            db_name = ids[i]
                            db_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
                            send_data_to_database(db_name, db_time)
                            self.confW = NotiWindow()    
                            self.timer.stop()
                            self.confW.show()
                            self.confW.setNoti(check,qImg)
                            self.confW.autoStart.connect(self.timer.start)
            def display_Image(self,frame):
                height = int(self.h / 2)
                width =int(self.w / 2)
                # print(height,width)
                height, width, channel = height,width,3
                # height, width, channel = frame.shape
                # cv2.putText(image,'check',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
                step = channel * width
                # create QImage from image
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                qImg = QImage(frame.data, width, height, step,QImage.Format_RGB888)
                # 
                # show image in img_label
                self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
                # self.ui.image_label.setScaledContents(1)
                return qImg
            
        #cua so hien ra khi cham cong (delay 5s giup nguoi dung confirm diem danh dung hay sai)
        class NotiWindow(QWidget):
            autoRecord = pyqtSignal()
            autoStart = pyqtSignal()
            def __init__(self,timeout=5):
                # call QWidget constructor
                super().__init__()
                self.ui = Noti_Form()
                self.ui.setupUi(self)

                self.ui.dongy_bt.clicked.connect(self.acceptContent)
                self.ui.chamlai_bt.clicked.connect(self.declineContent)
                self.time_to_wait = timeout
                self.timer = QtCore.QTimer(self)
                self.timer.setInterval(1000)
                self.timer.timeout.connect(self.changeContent)
                self.timer.start()

                # self.autoRecord.connect(self.close)
            def changeContent(self):
                self.ui.dongy_bt.setText("Đồng ý ({0})".format(self.time_to_wait))
                # self.setText("wait (closing automatically in {0} secondes.)".format(self.time_to_wait))
                if self.time_to_wait == 0:
                    self.close()
                    self.autoRecord.emit()
                    self.autoStart.emit() 
                    self.timer.stop()     
                self.time_to_wait -= 1
                print(self.time_to_wait)
            def acceptContent(self):
                self.close()
                self.autoRecord.emit()
                self.autoStart.emit() 
                self.timer.stop()  
            def declineContent(self):
                self.close()
                self.autoStart.emit() 
                self.timer.stop() 
            def setNoti(self,check,qImg):
                self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
                # self.ui.resize_image()
                # self.ui.image_label.setScaledContents(1)
                
                text = 'Họ và tên: '+check['name']
                self.ui.label.setText(text)
                self.ui.label.setWordWrap(True)
                self.ui.label.setText('\u200b'.join(text))
                text = 'Acc: ' + str(check['acc'])
                self.ui.label_2.setText(text)
                self.ui.label_2.setWordWrap(True)
                self.ui.label_2.setText('\u200b'.join(text))
                text = 'Thời gian: '+check['time'].strftime('%H:%M:%S')
                self.ui.label_3.setText(text)
                self.ui.label_3.setWordWrap(True)
                self.ui.label_3.setText('\u200b'.join(text))
                text = 'Ngày '+check['time'].strftime('%d')+ ', tháng '+check['time'].strftime('%m')+', năm '+ check['time'].strftime('%Y')
                self.ui.label_4.setText(text)
                self.ui.label_4.setWordWrap(True)
                self.ui.label_4.setText('\u200b'.join(text))
        if __name__ == '__main__':
            app = QApplication(sys.argv)
            # signal.signal(signal.SIGINT, lambda *args: QApplication.quit())
            # create and show mainWindow
            mainWindow = MainWindow()
            mainWindow.show()

            sys.exit(app.exec_())
    

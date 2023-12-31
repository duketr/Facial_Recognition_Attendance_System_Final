from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import run

import sys
sys.path.append('../')

import cv2
import numpy as np

import tensorflow as tf
import numpy as np
import re
import os
import cv2
from utils import face_preprocess
from utils.ROI_extraction import extract_ROI
import sklearn
import configparser

from nets.mtcnn_model import P_Net, R_Net, O_Net
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector

def load_mtcnn(conf):
    # load mtcnn model
    MODEL_PATH = conf.get("MTCNN", "MODEL_PATH")
    MIN_FACE_SIZE = int(conf.get("MTCNN", "MIN_FACE_SIZE"))
    STEPS_THRESHOLD = [float(i)  for i in conf.get("MTCNN", "STEPS_THRESHOLD").split(",")]

    detectors = [None, None, None]
    prefix = [MODEL_PATH + "/PNet_landmark/PNet",
              MODEL_PATH + "/RNet_landmark/RNet",
              MODEL_PATH + "/ONet_landmark/ONet"]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=MIN_FACE_SIZE, threshold=STEPS_THRESHOLD)

    return mtcnn_detector
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
def load_mobilefacenet(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.compat.v1.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            # print('TYPE-1')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.compat.v1.get_default_session(), os.path.join(model_exp, ckpt_file))
        # print('TYPE-2')
    inputs_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    sess = tf.compat.v1.Session()
    return sess, inputs_placeholder, embeddings
def load_faces(faces_dir, mtcnn_detector,sess,inputs_placeholder,embeddings):
    face_db = []
    
    # load_mobilefacenet("../models/mobilefacenet_model/mbfn_model.pb")
    inputs_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
    i=0
    for root, dirs, files in os.walk(faces_dir):
        for file in files:
            try:
                input_image = cv2.imread(os.path.join(root, file))
                faces, landmarks = mtcnn_detector.detect(input_image)
                
                
                bbox = faces[0,:4]
            except IndexError:
                print(f'Loi Anh {file} trong folder {root} khong chua mat nguoi')
                continue
            points = landmarks[0,:].reshape((5, 2))
            hist = extract_ROI(input_image, bbox)
            nimg = face_preprocess.preprocess(input_image, bbox, points, image_size='112,112')
            


            # cv2.imwrite(f'./face_detectDB/{i}.png',nimg)
            i+=1
            nimg = nimg - 127.5
            nimg = nimg * 0.0078125
            
            name = file.split(".")[0]
            # name = re.sub('[0-9]','',name)

            input_image = np.expand_dims(nimg,axis=0)
            
            feed_dict = {inputs_placeholder: input_image}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

            embedding = sklearn.preprocessing.normalize(emb_array).flatten()
            face_db.append({
                "name": name,
                "feature": embedding,
                "hist": hist
            })
    return face_db

# def load_faces(FACE_DB_PATH,sess, inputs_placeholder, embeddings):
#     # FACE_DB_PATH = FACE_DB_PATH
#     face_db = []
#     for root, dirs, files in os.walk(FACE_DB_PATH):
#         for file in files:
#             input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
#             try:
#                 input_image = input_image - 127.5
#                 input_image = input_image * 0.0078125
#                 name = file.split(".")[0]

#                 input_image = np.expand_dims(input_image, axis=0)

#                 feed_dict = {inputs_placeholder: input_image}
#                 emb_array = sess.run(embeddings, feed_dict=feed_dict)

#                 embedding = sklearn.preprocessing.normalize(emb_array).flatten()
#                 face_db.append({
#                     "name": name,
#                     "feature": embedding
#                 })
#                 print('loaded face: %s' % file)
#             except Exception as e:
#                 print(e)
#                 print("delete error image:%s" % file)
#                 os.remove(os.path.join(root, file))
#                 continue
#     return face_db
def add_faces(FACE_DB_PATH,TEMP_FACE_PATH,mtcnn_detector):
    face_db_path = FACE_DB_PATH
    faces_name = os.listdir(face_db_path)
    temp_face_path = TEMP_FACE_PATH
    for root, dirs, files in os.walk(temp_face_path):
        for file in files:
            if file not in faces_name:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                faces, landmarks = mtcnn_detector.detect(input_image)
                bbox = faces[0, :4]
                points = landmarks[0, :].reshape((5, 2))
                nimg = face_preprocess.preprocess(input_image, bbox, points, image_size='112,112')
                cv2.imwrite(os.path.join(face_db_path, os.path.basename(file)), nimg)

# MODEL_PATH = conf.get("MOBILEFACENET", "MODEL_PATH")
# VERIFICATION_THRESHOLD = float(conf.get("MOBILEFACENET", "VERIFICATION_THRESHOLD"))
# FACE_DB_PATH = conf.get("MOBILEFACENET", "FACE_DB_PATH")

# class runningBackGround():
#     def __init__(self):
#         self.mtcnn_detector = load_mtcnn(conf)
#         self.faces_db = load_facesdb(FACE_DB_PATH,self.mtcnn_detector)

# runningBackGround()
# print('load DONE')
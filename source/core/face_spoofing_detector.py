import cv2
import os
import glob
import numpy as np
import random
import tensorflow as tf
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support

# Method 1
# Using 14 features in
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.2807&rep=rep1&type=pdf


class FaceSpoofingSVM:

    def __init__(self, model_path="../models/weight_svm.txt"):
        self.model = joblib.load(model_path)

    def extract_14_features(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = np.zeros([14], np.float32)  # 14 feature and 1 label
        img_hat = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5, sigmaY=0.5)
        mn = (img.shape[0] * img.shape[1])  # m x n
        mse = np.sum(np.power((img - img_hat), 2)) / mn  # feature 1
        features[0] = mse
        psnr = 10 * np.log(np.max(img**2)) / mse  # feature 2
        features[1] = psnr
        snr = 10 * np.log(np.sum(img**2) / (mn * mse))  # feature 3
        features[2] = snr
        sc = np.sum(img**2) / np.sum(img_hat**2)  # feature 4
        features[3] = sc
        md = np.max(np.abs(img - img_hat))  # feature 5
        features[4] = md
        ad = np.sum(img - img_hat) / mn  # feature 6
        features[5] = ad
        nae = np.sum(np.abs(img - img_hat)) / np.sum(img)  # feature 7
        features[6] = nae
        diff = np.abs(img - img_hat).reshape([mn, -1])
        diff.sort()
        max_10 = diff[::-1][:10]
        ramd = 1 / 10 * np.sum(max_10)  # r = 10 # feature 8
        features[7] = ramd
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        blur_img = cv2.filter2D(img, -1, kernel)
        blur_img_hat = cv2.filter2D(img_hat, -1, kernel)
        diff_blur = (blur_img - blur_img_hat)**2
        blur_img_2 = (blur_img**2)
        lmse = np.sum(diff_blur[1:-1, 2:-1]) / np.sum(
            blur_img_2[1:-1, 2:-1])  # feature 9
        features[8] = lmse
        nxc = np.sum(img * img_hat) / np.sum(img**2)  # feature 10
        features[9] = nxc
        alpha = 2 / np.pi * np.arccos(
            np.dot(img, img_hat) /
            (np.linalg.norm(img) * np.linalg.norm(img_hat)))  # alpha
        mas = 1 - 1 / mn * np.sum(alpha)  # feature 11
        features[10] = mas
        mams = 1 / mn * np.sum(
            1 - (1 - alpha) *
            (1 - np.linalg.norm(img - img_hat) / 255))  # feature 12
        features[11] = mams
        IE_img = cv2.Sobel(np.float32(img), cv2.CV_32F, 1, 1)
        IE_img_hat = cv2.Sobel(np.float32(img_hat), cv2.CV_32F, 1, 1)
        ted = np.sum(np.abs(IE_img - IE_img_hat)) / mn  # feature 13
        features[12] = ted
        # count corners
        corners_img = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
        corners_img_hat = cv2.goodFeaturesToTrack(img_hat, 25, 0.01, 10)
        tcd = np.abs(len(corners_img) - len(corners_img_hat)) / max(
            len(corners_img), len(corners_img_hat))  # feature 14
        features[13] = tcd
        return features

    def is_face_spoofing(self, img):
        clone = img
        feature = np.array([self.extract_14_features(clone)])
        label = self.model.predict(feature)
        if label == 0:
            return True
        else:
            return False


class FaceSpoofingModel3:

    def __init__(self, model_path="../models/method3_model.h5"):
        self.model = load_model(model_path)
        self.graph = tf.get_default_graph()

    def is_face_spoofing(self, img):
        clone = img
        clone = cv2.resize(clone, (160, 160))
        clone = np.reshape(clone, [1, 160, 160, 3])
        with self.graph.as_default():
            label = np.argmax(self.model.predict(clone))
        if label == 0:
            return True
        else:
            return False

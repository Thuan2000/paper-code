import cv2
import numpy as np
from timeit import time
from config import Config

class BGSProcess():
    def __init__(self, history=5, lr=0.0005,
                 backgroundratio=0.9, shadowthresh=0.9,
                 object_size=100, detectshadow_flag=True,
                 varthreshold=16, rect_size=(3, 3),
                 ellipse_size=(3, 3), high_rect_size=(3, 10)):
        self.lr = lr
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=detectshadow_flag)
        self.fgbg.setBackgroundRatio(backgroundratio)
        self.fgbg.setShadowThreshold(shadowthresh)
        self.fgbg.setVarThreshold(varthreshold)
        self.Rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, rect_size)
        self.Ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ellipse_size)
        self.heightRect_kernel = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, high_rect_size)
        self.object_size = object_size

    def process(self, frame, ROI_Cor=None):
        ''' ROI_cor is a list of tuple, each tuple is a (x,y) cordinator of each polygon point
            if u dont want to use ROI, pass a None to ROI_cor
            Output is a list of boxes with tlbr attribute'''
        self.boxes = []
        self.frame = frame
        self.frame = cv2.GaussianBlur(self.frame, (5,5),1)
        self.fgmask = self.fgbg.apply(self.frame, learningRate = self.lr)
        self.sum_cols = np.sum(self.fgmask, axis = 0)/255
        self.sum_rows = np.sum(self.fgmask, axis = 1)/255
        self.fgmask[self.fgmask < 255] = 0
        self.fgmask = cv2.erode(self.fgmask ,self.Rect_kernel, iterations = 3)
        self.fgmask = cv2.dilate(self.fgmask, self.Ellip_kernel, iterations = 2)
        if ROI_Cor is not None:
            self.black_mask = np.zeros(self.frame.shape[:2],dtype = np.int8)
            cv2.fillConvexPoly(self.black_mask,np.array(ROI_Cor),255)
            self.fgmask[self.fgmask * self.black_mask == 0] = 0
        self.fgmask = cv2.morphologyEx(self.fgmask, cv2.MORPH_CLOSE, self.heightRect_kernel, iterations = 3)
        self.fgmask = cv2.dilate(self.fgmask, self.Ellip_kernel, iterations = 2)
        im2, self.contours, hierarchy = cv2.findContours(self.fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctn = sorted(self.contours, key = cv2.contourArea)
        for c in self.contours:
            if cv2.contourArea(c) < self.object_size:
                continue
            (x,y,w,h) = cv2.boundingRect(c)
            self.boxes.append([x, y, x+w, y+h])
        return self.boxes
    
    def preprocess(self, image):
        ''' preprocess image before put in yolo '''
        frame = cv2.GaussianBlur(image, (5,5), 1)
        self.fgmask = self.fgbg.apply(image, learningRate=self.lr)
        self.fgmask = cv2.erode(self.fgmask ,self.Rect_kernel, iterations = 3)
        self.fgmask = cv2.dilate(self.fgmask, self.Ellip_kernel, iterations = 2)
        self.fgmask = cv2.morphologyEx(self.fgmask, cv2.MORPH_OPEN, self.heightRect_kernel, iterations = 3)
        im2, self.contours, hierarchy = cv2.findContours(self.fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ctn = sorted(self.contours, key = cv2.contourArea)
        if len(ctn)>0:
            (x,y,w,h) = cv2.boundingRect(ctn[-1])
            if w*h*3 > Config.Preprocess.BGS_RATIO*image.size:
                return True
            else :
                return False
        return False

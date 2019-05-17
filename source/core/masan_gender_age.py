import cv2
import sys
import numpy as np
import datetime
import mxnet as mx
from skimage import transform as trans
from core.gender_age import face_model

class GenderAgeModel():

    def __init__(self, imagesize, modelpath, gpu_id =0):
        self.model = face_model.FaceModel(imagesize, modelpath, gpu_id)
        print('init model gender age', "="*10)

    def preprocessing( img, landmark=None, **kwargs):
        img = img
        M = None
        image_size = []
        str_image_size = kwargs.get('image_size', '')
        if len(str_image_size)>0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size)==1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size)==2
            assert image_size[0]==112
            assert image_size[0]==112 or image_size[1]==96
        if landmark is not None:
            assert len(image_size)==2
            src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32 )
            if image_size[1]==112:
                src[:,0] += 8.0
            dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            M = tform.params[0:2,:]

            warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
            return warped
        return img

    def get_gender_age(self, img, landmark = None):
        nimg = GenderAgeModel.preprocessing(img, landmark, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        image = mx.io.DataBatch(data=(data,))
        gender, age = self.model.get_ga(image)
        list_ga = [gender, age]
        list_ga = list(map(int,list_ga))

        return list_ga
from frame_reader import URLFrameReader
from cv_utils import show_frame
from face_detector import MTCNNDetector
from tf_graph import FaceGraph

#import glasses_mask
import mask_glasses

from cv_utils import CropperUtils
from frame_process import ROIFrameProcessor
import cv2
from preprocess import Preprocessor, normalization
import numpy as np
import click
from scipy import misc
from config import Config
from cv_utils import create_if_not_exist
import time
import pipe

frame_reader = URLFrameReader(0)
face_detector = MTCNNDetector(FaceGraph())
frame_processor = ROIFrameProcessor(scale_factor=2)

mask_classifier = mask_glasses.MaskClassifier()
glasses_classifier = mask_glasses.GlassesClassifier()

preprocessor = Preprocessor(algs=normalization)

MASK_DIR = '%s/data/Mask/' % Config.ROOT
NOMASK_DIR = '%s/data/No_Mask/' % Config.ROOT
GLASSES_DIR = '%s/data/Glasses/' % Config.ROOT
NOGLASSES_DIR = '%s/data/No_Glasses/' % Config.ROOT

create_if_not_exist(MASK_DIR)
create_if_not_exist(NOMASK_DIR)
create_if_not_exist(GLASSES_DIR)
create_if_not_exist(NOGLASSES_DIR)

while frame_reader.has_next():
    frame = frame_reader.next_frame()
    if frame is None:
        break

    bbs, pts = face_detector.detect_face(frame)
    preprocessed = preprocessor.process(frame)

    has_mask = None
    has_glasses = None

    if len(bbs) > 0:
        # cropped = CropperUtils.crop_face(frame, bbs[0], return_size=224)
        has_mask = mask_classifier.is_wearing_mask(preprocessed)[0]
        has_glasses = glasses_classifier.is_wearing_glasses(preprocessed)[0]
        show_frame(
            frame,
            bb=bbs[0],
            pt=pts[:, 0],
            id='Has mask %s, has_glasses: %s' % (has_mask, has_glasses))
        #if has_glasses == True: print("Allert!!!!!!!!!!!!!!!!")
    else:
        show_frame(frame)

#    key = click.getchar()

#    print('Has mask %s, has_glasses: %s' % (has_mask, has_glasses))

#    if key == 'a':
#    	if has_mask:
#   		# no mask, but detect has mask
#    		file_name = '%s/%s.jpg' % (NOMASK_DIR, time.time())
#    	else:
#    		# has mask, but detect no mask
#    		file_name = '%s/%s.jpg' % (MASK_DIR, time.time())
#    	misc.imsave(file_name, frame)

#    if key == 'a':
#    	if has_glasses:
#		# no glasses, but detect has glasses
#    		file_name = '%s/%s.jpg' % (NOGLASSES_DIR, time.time())
#    	else:
# has glasses, but detect no glasses
#    		file_name = '%s/%s.jpg' % (GLASSES_DIR, time.time())
#    	misc.imsave(file_name, frame)
#    else:
#    	show_frame(frame)

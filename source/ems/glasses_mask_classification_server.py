import os
import cv2
import time
import numpy as np
from ems import base_server
from core import tf_graph, face_detector, preprocess
from core import mask_glasses
from core.cv_utils import CropperUtils
import config as Config


class GlassesMaskClassificationServer(base_server.AbstractServer):

    def init(self):
        self.bind_dir = os.environ.get('BIND_DIR', '')
        # below is for mask_glass_classification
        self.gm_face_detector = face_detector.MTCNNDetector(tf_graph.FaceGraph())
        self.gm_preprocessor = preprocess.Preprocessor(algs=preprocess.normalization)
        self.gm_glasses_classifier = mask_glasses.GlassesClassifier()
        self.gm_mask_classifier = mask_glasses.MaskClassifier()

    def add_endpoint(self):
        self.app.add_url_rule('/glasses-mask-classification', 'glasses-mask-classification', self.glasses_mask_classification_api, methods=['POST'])

    def glasses_mask_classification_api(self):
        images = self.input_images_parser()
        results = []
        for image in images:
            if image is not None:
                image_cvted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bbs, pts = self.gm_face_detector.detect_face(image_cvted)
                if len(bbs) > 0:
                    # print(bbs)
                    preprocessed_image = self.gm_preprocessor.process(image)
                    preprocessed_image = np.array(preprocessed_image)
                    has_masks = self.gm_mask_classifier.is_wearing_mask(preprocessed_image)
                    has_glasses = self.gm_glasses_classifier.is_wearing_glasses(preprocessed_image)
                    results.append((has_glasses[0], has_masks[0]))
            else:
                results.append((None, None))

        print('predictions', results)
        if results:
            message = 'Format: [(has_glasses, has_mask), ...]'
        else:
            message = 'Can not save image, please try another one'
        return self.response_success({'predictions': results}, message=message)

    def input_images_parser(self):
        images_list = []
        # print(self.request.form)
        # handle relative path in bd2
        if 'images[]' in self.request.form:
            images_list_str = self.request.form.getlist('images[]')
            for image_path in images_list_str:
                print('reading', image_path)
                # extract cordinates from file name
                image_path = os.path.join(self.bind_dir, image_path.rstrip())
                image = cv2.imread(image_path)
                if (image is not None) and (image.size > 0):
                    images_list.append(image)
                else:
                    images_list.append(None)
        # handle images file
        files_dict = self.request.files.to_dict(flat=False)
        # print(files_dict.keys(), files_dict.values())
        if 'images[]' in files_dict:
            images = files_dict['images[]']
            # print(images)
            for image in images:
                save_path = os.path.join(os.getcwd(), image.filename)
                image.save(save_path)
                _image = cv2.imread(save_path)
                images_list.append(_image)
                os.remove(save_path)

        return images_list

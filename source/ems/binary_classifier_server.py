import os
import cv2
import time
from ems import base_server
from core import binary_classifier
from core.cv_utils import CropperUtils


class MasanCustomerClassificationServer(base_server.AbstractServer):

    def __init__(self):
        self.classifier = binary_classifier.MasanCustomerClassification(use_cuda=True)
        self.bind_dir = os.environ.get('BIND_DIR', '')
        super(MasanCustomerClassificationServer, self).__init__()

    def add_endpoint(self):
        self.app.add_url_rule('/classify', 'classify', self.classify, methods=['POST'])

    def classify(self):
        images_array_list = self.input_images_parser()
        if len(images_array_list) == 0:
            return self.response_error('No image in track list')
        predictions = []
        for image in images_array_list:
            if image is not None:
                predictions.append(self.classifier.predict(image))
            else:
                predictions.append(None)
        # assert len(staff_flag_list) == len(images_array_list),'number of output flags not match with number of input images'
        print('predictions', predictions)
        return self.response_success({'predictions': predictions})

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
                try:
                    bb_cordinate = image_path.split('.')[-2].split('_')[-4:]
                    bb_cordinate = '_'.join(bb_cordinate)
                    image = cv2.imread(image_path)
                    if (image is not None) and (image.size > 0):
                        cropped_image = CropperUtils.reverse_display_face(image, bb_cordinate)
                        images_list.append(cropped_image)
                    else:
                        images_list.append(None)
                except (IndexError, ValueError):
                    images_list.append(None)
                    continue
        # handle images file
        return images_list

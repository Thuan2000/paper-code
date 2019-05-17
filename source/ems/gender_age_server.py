import os
import cv2
import time
import numpy as np
from ems import base_server
from core import masan_gender_age
from config import Config


class GenderAgeServer(base_server.AbstractServer):

    # 0 - Female, 1 - Male
    def __init__(self):
        self.gender_age = masan_gender_age.GenderAgeModel(imagesize='112,112', modelpath=Config.Model.GENDER_AGE_DIR, gpu_id=0)
        self.bind_dir = os.environ.get('BIND_DIR', '')
        super(GenderAgeServer, self).__init__()

    def add_endpoint(self):
        self.app.add_url_rule('/gender-age', 'gender-age', self.gender_age_api, methods=['POST'])

    def gender_age_api(self):
        images_array_list = self.input_images_parser()
        if len(images_array_list) == 0:
            return self.response_error('No image in track list')
        gender_age_list = []
        for image, landmark in images_array_list:
            if image is not None:
                gender_age_list.append(self.gender_age.get_gender_age(image, landmark))
            else:
                gender_age_list.append([None, None])

        print('predictions', gender_age_list)
        return self.response_success({'predictions': gender_age_list})

    def input_images_parser(self):
        images_list = []
        # print(self.request.form)
        if 'images[]' in self.request.form:
            images_list_str = self.request.form.getlist('images[]')
            for image_path_str in images_list_str:
                # extract cordinates from file name
                image_path_str = os.path.join(self.bind_dir, image_path_str.rstrip())
                image_path , landmark = image_path_str.split(',')
                bb_cordinate = image_path.split('.')[-2].split('_')[-4:]
                bb_cordinate = '_'.join(bb_cordinate)
                landmark = np.array(landmark.split('_')).reshape(2,5).T
                image = cv2.imread(image_path)
                print('reading', image_path)
                if (image is not None) and (image.size > 0):
                    images_list.append([image, landmark])
                else:
                    images_list.append([None,None])
        return images_list


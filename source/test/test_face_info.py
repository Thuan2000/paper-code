import unittest
import math
import numpy as np
from face_info import FaceInfo


class TestFaceInfo(unittest.TestCase):

    def setUp(self):
        bounding_box = np.random.randint(100, size=(4))
        frame_id = 1
        face_image = np.array((160,160))
        str_padded_bbox = '_'.join(bounding_box.astype('str').tolist())
        landmarks = np.random.randint(100, size=(10))
        self.face_info = FaceInfo(bounding_box, frame_id, face_image,
                                  str_padded_bbox, landmarks)

    def test_it_should_has_default_attributes_values(self):
        ''' face info should sucessfully assign it's default attributes '''
        self.assertIsNone(self.face_info.embedding)
        self.assertIsNone(self.face_info.image_id)
        self.assertIsInstance(self.face_info.quality, int)

    def test_update_embedding_should_updated_values_into_attribute(self):
        ''' input random int or np.array to test if embedding is assigned into the object '''
        rand_int = np.random.randint(10)
        self.face_info.update_embedding(rand_int)
        self.assertEqual(self.face_info.embedding, rand_int)
        rand_array = np.random.random((128))
        self.face_info.update_embedding(rand_array)
        self.assertTrue(np.array_equal(self.face_info.embedding, rand_array))

    def test_set_face_quality_should_updated_values_into_attribute(self):
        ''' input random int to test if face quality is assigned into the object '''
        rand_int = np.random.randint(10)
        self.face_info.set_face_quality(rand_int)
        self.assertEqual(self.face_info.quality, rand_int)

    def test_is_good_should_handle_boundaries_values_without_failing(self):
        ''' input infinite value for yaw, pitch, face quality to test durability '''
        values = [np.inf, 0, -np.inf]
        for value in values:
            self.face_info.yaw_angle = value
            self.face_info.pitch_angle = value
            self.face_info.quality = value
            self.assertIsInstance(self.face_info.is_good(), bool)

    def test_is_good_return_should_true_on_true_testcases(self):
        ''' predefine true test case for is good and check if the result is true '''
        self.face_info.yaw_angle = 0
        self.face_info.pitch_angle = 0
        self.face_info.quality = np.inf
        self.assertTrue(self.face_info.is_good())

    def test_is_good_return_should_fail_on_false_testcases(self):
        ''' branch test is_good on false testcases with expecting output to return false '''
        yaw_values = [0, 0, 0, 0, 800, 800, 800, 800]
        pitch_values = [800, 800, 0, 0, 800, 800, 0, 0]
        face_qualities = [0, 20, 0, 0, 20, 0, 20, 0]
        for y,p,fq in zip(yaw_values, pitch_values, face_qualities):
            self.face_info.yaw_angle = y
            self.face_info.pitch_angle = p
            self.face_info.quality = fq
            self.assertFalse(self.face_info.is_good())

    def test_str_info_should_return_a_string(self):
        ''' this method return a string of frame_id'''
        self.assertIsInstance(self.face_info.str_info(), str)

    def test_str_info_should_include_the_face_info_attributes(self):
        ''' input differences face_info attribute and check if those are in return string '''
        self.face_info.frame_id = 5
        self.face_info.str_padded_bbox = 'random_txt_string'
        output_str = self.face_info.str_info()
        self.assertIn('5', output_str)
        self.assertIn('random_txt_string', output_str)

    def test_angle_between_two_vector_should_return_a_float(self):
        ''' mock a random pairs of vector to check return type '''
        vector_1 = np.random.random(10)
        vector_2 = np.random.random(10)
        angle = self.face_info.angle_between(vector_1, vector_2)
        self.assertIsInstance(angle, float)

    def test_angle_between_result_return_angle(self):
        ''' input two pre-calculated vecetor to get expected result '''
        vector_1 = np.array([1, 2 , 3])
        vector_2 = np.array([4, 5, 6])
        angle = round(self.face_info.angle_between(vector_1, vector_2), 3)
        self.assertEqual(angle, 0.226)

    def test_calc_face_angle_should_run_on_every_branch(self):
        ''' input different points to test calc_face_angle_conditional_branch '''
        points = np.random.randint(100, size=(10))
        points[0] = 1
        points[2] = 0
        self.assertIsInstance(self.face_info.calc_face_angle(points), float)
        points = np.random.randint(100, size=(10))
        points[0] = 0
        points[2] = 1
        self.assertIsInstance(self.face_info.calc_face_angle(points), float)
        points = np.random.randint(100, size=(10))
        points[1] = 1
        points[2] = 0       
        self.assertIsInstance(self.face_info.calc_face_angle(points), float)
        points = np.random.randint(100, size=(10))
        points[0] = 1
        points[2] = 0      
        self.assertIsInstance(self.face_info.calc_face_angle(points), float)

    def test_calc_face_pitch_should_run_on_every_branch(self):
        ''' input different points to test calc_face_angle_conditional_branch '''
        points = np.random.randint(100, size=(10))
        points[0] = 1
        points[2] = 0       
        self.assertIsInstance(self.face_info.calc_face_pitch(points), float)
        points = np.random.randint(100, size=(10))
        points[0] = 0
        points[2] = 1        
        self.assertIsInstance(self.face_info.calc_face_pitch(points), float)
        points = np.random.randint(100, size=(10))
        points[1] = 1
        points[2] = 0      
        self.assertIsInstance(self.face_info.calc_face_pitch(points), float)
        points = np.random.randint(100, size=(10))
        points[0] = 1
        points[2] = 0
        self.assertIsInstance(self.face_info.calc_face_pitch(points), float)

    def test_deep_clone_is_copy_values_but_nott_object_itself(self):
        ''' clone a object and check for every coppied attributes is have the same value but of a reference '''
        cloned_face_id = self.face_info.deep_clone()
        self.assertTrue(np.array_equal(cloned_face_id.bounding_box, self.face_info.bounding_box))
        self.assertTrue(np.array_equal(cloned_face_id.embedding, self.face_info.embedding))
        self.assertEqual(cloned_face_id.frame_id, self.face_info.frame_id)
        self.assertTrue(np.array_equal(cloned_face_id.face_image, self.face_info.face_image))
        self.assertEqual(cloned_face_id.str_padded_bbox, self.face_info.str_padded_bbox)
        self.assertEqual(cloned_face_id.time_stamp, self.face_info.time_stamp)
        self.assertTrue(np.array_equal(cloned_face_id.landmarks, self.face_info.landmarks))
        self.assertEqual(cloned_face_id.image_id, self.face_info.image_id)
        self.assertEqual(cloned_face_id.quality, self.face_info.quality)
        self.assertEqual(cloned_face_id.yaw_angle, self.face_info.yaw_angle)
        self.assertEqual(cloned_face_id.pitch_angle, self.face_info.pitch_angle)

if __name__ == '__main__':
    unittest.main()
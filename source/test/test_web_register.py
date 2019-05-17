from web_register import sort_filenames, register_function
from face_detector import MTCNNDetector
from face_extractor import FacenetExtractor
from tf_graph import FaceGraph
from config import Config


def test_sort_filenames(txts):
    result = sort_filenames(txts)
    print(result)
    print('Sort_filenames OK')


def test_register_function():
    face_rec_graph = FaceGraph()
    face_extractor = FacenetExtractor(
        face_rec_graph, model_path=Config.FACENET_DIR)
    detector = MTCNNDetector(face_rec_graph)
    preprocessor = Preprocessor()
    register_function(
        detector,
        preprocessor,
        face_extractor,
    )


test_sort_filenames(txts_images)

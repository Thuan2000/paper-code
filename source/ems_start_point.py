'''
Decide with ems service will run when container start
'''
import argparse
import os
from core.cv_utils import create_if_not_exist
from utils.logger import logger
from config import *



if __name__ == '__main__':
    # Main steps
    service_type = os.environ.get('SERVICE_TYPE', SERVICE_ANNOTATION)
    classification_case = os.environ.get('CLASSIFICATION_CASE')
    print('Service type', service_type)

    if service_type == SERVICE_ANNOTATION:
        print('Run annotation server')
        container_id = os.environ.get('CV_SERVER_NAME')
        subcription_id = os.environ.get('SUBSCRIPTION_ID')
        from ems import annotation_server
        server = annotation_server.AnnotationServer(subcription_id)
        server.run()

    elif service_type == SERVICE_DEMO_REGISTER:
        print('Run demo server')
        from ems import demo_register_server
        server = demo_register_server.DemoRegisterServer()
        server.run()

    elif service_type == SERVICE_ECOMMERCE:
        print('Run demo ecommerce')
        from ems import demo_ecommerce_server
        server = demo_ecommerce_server.DemoEcommerceServer()
        server.run()

    elif service_type == SERVICE_MULTICAMERA:
        print('Run demo unilever')
        camera_id = os.environ.get('CAMERA_ID')
        from prod.unilever_alert import server as uni_server
        server = uni_server.UnileverServer(camera_id)
        server.run()

    elif service_type == SERVICE_RETENTION_FACE:
        print('Run retention face production')
        from ems import retention_server_face
        server = retention_server_face.RetentionServer()
        server.run()

    elif service_type == SERVICE_RETENTION_TRAFFIC:
        print('Run retention traffic production')
        from ems import retention_server_traffic
        server = retention_server_traffic.RetentionServer()
        server.run()

    elif service_type == SERVICE_ATM_AUTHENTICATION:
        print('Run atm authentication server')
        from ems import atm_authentication_server
        server = atm_authentication_server.ATMAuthenticationServer()
        server.run()

    elif service_type == SERVICE_FACE_VERIFICATION:
        print('Run face verification server')
        from ems import face_verification_server
        server = face_verification_server.FaceVerificationServer()
        server.run()

    elif service_type == SERVICE_MASSAN_STAFF_CLASSIFICATION:
        print('Run service micro classification')
        from ems import binary_classifier_server
        server = binary_classifier_server.MasanCustomerClassificationServer()
        server.run()

    elif service_type == SERVICE_AGE_GENDER_PREDICTION:
        print('Run service age and gender prediction')
        from ems import gender_age_server
        server = gender_age_server.GenderAgeServer()
        server.run()

    elif service_type == SERVICE_GLASSES_MASK_CLASSIFICATION:
        print('Run service glasses and mask classification')
        from ems import glasses_mask_classification_server
        server = glasses_mask_classification_server.GlassesMaskClassificationServer()
        server.run()
    elif service_type == SERVICE_VERIFICATION_CMND:
        print('Run service chung minh nhan dan verification')
        from ems import cmnd_verification_server.CmndVerificationServer()
    elif service_type == SERVICE_DASHBOARD:
        print('TODO: Start dashboard server')

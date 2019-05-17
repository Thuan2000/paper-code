'''
Wrapper around all configuration
To use, first add configuration dir to python path
export PYTHONPATH=
export PYTHONPATH=$(pwd)/configuration:$PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
'''
import os

SERVICE_DEFAULT = 'default'
SERVICE_ANNOTATION = 'annotation'
SERVICE_DEMO_REGISTER = 'face-recognition'
SERRVICE_DEMO_CLASSIFICATION = 'classification'
SERVICE_DASHBOARD = 'dashboard'
SERVICE_ECOMMERCE = 'ecommerce'
SERVICE_MULTICAMERA = 'multicam_monitor'
SERVICE_RETENTION_FACE = 'retention-face'
SERVICE_RETENTION_TRAFFIC = 'retention-traffic'
SERVICE_ATM_AUTHENTICATION = 'atm_authentication'
SERVICE_FACE_VERIFICATION = 'face_verification'
SERVICE_MASSAN_STAFF_CLASSIFICATION = 'massan_staff_classification'
SERVICE_AGE_GENDER_PREDICTION = 'age_gender_prediction'
SERVICE_GLASSES_MASK_CLASSIFICATION = 'glasses_mask_classification'
SERVICE_VERIFICATION_CMND = 'verification-cmnd'


service_type = os.environ.get('SERVICE_TYPE', SERVICE_DEFAULT)

if service_type == SERVICE_ANNOTATION:
    import configuration.annotation as Config
    print('Use annotation config')
elif service_type == SERVICE_DEMO_REGISTER:
    import configuration.demo_register as Config
    print('Use demo config')
elif service_type == SERVICE_VERIFICATION_CMND:
    import configuration.verification_cmnd
elif service_type == SERVICE_DASHBOARD:
    print('Use dashboard config')
elif service_type == SERRVICE_DEMO_CLASSIFICATION:
    print('Use classification config')
    import configuration.demo_glasess_mask as Config
elif service_type == SERVICE_ECOMMERCE:
    print('Use ecommerce config')
    import configuration.demo_ecommerce as Config
elif service_type == SERVICE_MULTICAMERA:
    print('Use multicam_monitor config')
    import prod.unilever_alert.config as Config
elif service_type == SERVICE_ATM_AUTHENTICATION:
    print('Use atm authentication config')
    import configuration.atm_authentication as Config
elif service_type == SERVICE_FACE_VERIFICATION:
    print('Use SERVICE_FACE_VERIFICATION config')
    import configuration.face_verification as Config
elif service_type == SERVICE_RETENTION_FACE:
    print('Use retention face config')
    import configuration.retention_dashboard_face as Config
elif service_type == SERVICE_RETENTION_TRAFFIC:
    print('Use retention traffic config')
    import configuration.retention_dashboard_traffic as Config
else:
    import configuration.base_config as Config
    print('Use default config')

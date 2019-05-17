from configuration.base_config import *
import configuration.base_config as Config


class MongoDB(Config.MongoDB):
    IP_ADDRESS = 'mongod-unilever'
    DB_NAME = 'multicam-dashboard'
    ALERT_COL = 'multicam_alert'
    CAMERA_COL = 'multicam_camera'
    SNAPSHOT_COL = 'multicam_snapshot'


class Socket:
    HOST = 'webapp-unilever-dashboard_webapp_1'
    PORT = 3000
    NAMESPACE = r'/multicam'
    ALERT_EVENT = 'new_alert'
    STATISTIC = 'process_info_update'


class Violations:
    TRESPASSING = 'trespassing'


class Dir(Config.Dir):
    VIDEO= 'video'
    SNAPSHOT = 'snapshot'


class ViolationAlert:
    BUFFER = 10
    NROF_SNAPSHOT = 10

class LogFile:
    LOG_NAME = os.environ.get('CV_SERVER_NAME', 'multicamera')
    LOG_FILE = os.path.join(Dir.LOG_DIR, '%s.log' % LOG_NAME)


class MOG:
    LR =0.0005
    HISTORY = 5
    BACKGROUNDRATIO = 0.9
    SHADOWTHRESH = 0.9
    VARTHRESHOLD = 16
    FRAME_SIZE = (1600, 800)
    OBJECT_SIZE = 100
    RECT_SIZE = (3, 3)
    ELLIPSE_SIZE = (3, 3)
    HIGH_RECT_SIZE = (3,10)
    DETECTSHADOW_FLAG = False


ACTION_TYPE = ['Co nguoi o vung cam',
                ' Nuoc tran',
                ' Vuot rao',
                'Nem do qua hang rao',
                ' Hut thuoc, nghe dien thoai',
                'xe di nguoc chieu',
                'xe dau trong vung cam',
                'Nguoi di bo trong vung chi danh cho xe']

from cv_utils import refresh_folder
from config import Config
from pymongo import MongoClient
import argparse


def main(web_mdb):
    mongodb_client = MongoClient(
        Config.MongoDB.IP_ADDRESS,
        Config.MongoDB.PORT,
        username=Config.MongoDB.USERNAME,
        password=Config.MongoDB.PASSWORD)

    mongodb_db = mongodb_client[Config.MongoDB.DB_NAME]
    mongodb_dashinfo = mongodb_db[Config.MongoDB.DASHINFO_COLS_NAME]
    mongodb_faceinfo = mongodb_db[Config.MongoDB.FACEINFO_COLS_NAME]
    mongodb_mslog = mongodb_db[Config.MongoDB.MSLOG_COLS_NAME]

    mongodb_client.drop_database(web_mdb)
    mongodb_client.admin.command('copydb', fromdb='web-backup', todb=web_mdb)

    #wipe data
    mongodb_dashinfo.remove({})
    mongodb_faceinfo.remove({})
    mongodb_mslog.remove({})
    refresh_folder(Config.TRACKING_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-wdb',
        '--web_mdb',
        help='web mongodb default: ' + Config.MAIN_NAME + '-dashboard',
        default=Config.MAIN_NAME + '-dashboard')

    args = parser.parse_args()
    main(args.web_mdb)

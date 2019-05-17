from database import AnnotationDatabase
import argparse
from bson.objectid import ObjectId

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Test only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', help='Storage path', default=None)
    args = parser.parse_args()
    db = AnnotationDatabase()
    dataset_id = db.mongodb_dataset.find_one({'storagePath': args.path})['_id']
    faceIds = []
    cursors = db.mongodb_image.find({'dataset': dataset_id})
    for cursor in cursors:
        faceIds.append(cursor['faceId'])
    faceIds = set(faceIds)
    print(faceIds)
    print(len(faceIds))

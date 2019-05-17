import os

class TimestampGetter():
    def nas_timestamp(self, filepath):
        filename = os.path.basename(filepath)
        timestamp = filename.split('.')[0].split('-')[4]
        return int(timestamp)
import requests
import json
import os


class HTTPRequest():

    def __init__(self, url):
        self.url = url

    def post_list(self, key, image_list):
        # payload = {key : json.dumps(data)}
        data = [(key, i) for i in image_list]
        try:
            response = requests.post(self.url, data=data)
            return response
        except:
            return None


class TrackerHTTPRequest():

    def __init__(self, url, **kwargs):
        self.url = url
        self.path_prefix = kwargs.get('path_prefix', '')

    def post_list(self, key, image_list):
        data = []
        for image_id in image_list:
            server_relative_path = os.path.join(self.path_prefix, image_id)
            data.append((key, server_relative_path))
        try:
            response = requests.post(self.url, data=data)
            if response.status_code == 200:
                json_data = response.json()
                if 'data' in json_data:
                    return json_data['data']
        except:
            pass
        return None

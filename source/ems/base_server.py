from flask import Flask, request, jsonify


class AbstractServer():

    def __init__(self):
        self.app = Flask(__name__)
        self.request = request
        self.add_endpoint()
        self.init()

    def init(self):
        # subclass do building pipeline, ... in here
        pass

    def add_endpoint(self):
        # subclass add api endpoint in this function
        pass

    def run(self):
        self.app.run(host='0.0.0.0', port=9001)

    @staticmethod
    def response_success(result, **kwargs):
        message = kwargs.get('message')
        if message is not None:
            return jsonify({'status': "successful", 'data': result, 'message': message})
        return jsonify({'status': "successful", 'data': result})

    @staticmethod
    def response_error(message):
        return jsonify({'message': message, 'status': 'failed'})

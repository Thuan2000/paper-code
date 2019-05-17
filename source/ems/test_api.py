from flask import Flask, request
from ems import base_server
import pdb


class MyFlaskApp(base_server.AbstractServer):

    def init(self):
        print('init')

    def add_endpoint(self):
        self.app.add_url_rule('/test', 'test', self.test, methods=['POST'])

    def test(self):
        return MyFlaskApp.response_success('Ok')


app = MyFlaskApp()
app.run()

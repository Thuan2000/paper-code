from flask import Flask

app = Flask(__name__)


@app.before_first_request
def run_before():
    print('Run before')


@app.route('/upload')
def upload():
    print('Upload')
    return "Ok"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)

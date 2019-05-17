# Face Recognition at EyeQ

## Bugs and debug
- Pika:
	- AttributeError: 'NoneType' object has no attribute 'poll'
	- self._poll.modify(fileno, events)
	- FileNotFoundError: [Errno 2] No such file or directory
	- pip install pika==0.12.0b2

## Repo structure

- models: contains all pre-trained models
	- align
	- detection

- data: toy examples that will be used for testing basic modules (will not exceed 50MB)

- src: contains all source code
	all main scripts are in first level
	- onetime: scripts that will be run once in a while (mostly for training hyperparameters)
	- test: scripts to test big modules that requires loading data
	- prod: scripts to be used in production for different use cases, prefixed with clients' names


## Linter
use flake8 that includes pyflakes, pep8 and McCabe code convention
https://www.smallsurething.com/how-to-automatically-lint-your-python-code-on-commit/
run this at main dir level
flake8

(will be run during git commit)


## Settings for Sublm users
Go to Preferences -> Settings
{
	"font_size": 16,
	"draw_white_space": "all",
	"tab_size": 4,
	"translate_tabs_to_spaces": false,
	"trim_automatic_white_space": true
}


## Doctest
Essentially all methods need to be accompanied with doctest


## Run test
./run_all_tests.sh

## Build docker for deployment
change url in run_eyeq.sh
nvidia-docker build -t cv-server .

## Run docker
nvidia-docker run \
	-v $(pwd)/session/:/home/eyeq/session \
	-v $(pwd)/data:/home/eyeq/data \
	-v $(pwd)/models:/home/eyeq/models \
	-v $(pwd)/src:/home/eyeq/src \
	--network host \
	-it cv-server
#### Inside docker image
./run_eyeq.sh

# TODO: Fill in this README


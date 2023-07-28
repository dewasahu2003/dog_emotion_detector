install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C,W1203,W0702 ./model/*.py

format:
	black ./model/*.py


all: install lint format
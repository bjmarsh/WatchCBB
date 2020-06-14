#! /bin/bash

sudo apt-get update
sudo apt-get install nginx gunicorn3 python3-pip python3-flask libpq-dev
pip3 install -r requirements.txt

#! /bin/bash

sudo apt-get update
sudo apt-get install nginx gunicorn3 python3-pip python3-flask libpq-dev postgresql
pip3 install -r requirements.txt  # text file with all of your required python packages

#####################
# Some useful things
#####################
#
# Copy a file/directory from local computer
# >>> scp -i <your_key.pem> -r <local file> <username>@<ec2_ip_address>:<desired_destination_on_ec2>
#   (-r only necessary if copying a directory)
#
#
# Running jupyter-notebook from EC2 instance
# SSH with this command:
# >>> ssh -i <your_key.pem>  -L localhost:4444:localhost:4444 <username>@<ec2_ip_address>
# Now on EC2 instance:
# >>> pip3 install jupyter
# >>> jupyter-notebook --no-browser --port=4444 --ip=127.0.0.1
# This should print out a URL that you can go to and use as normal
#
#
# Installing/configuring postgresql:
# >>> sudo apt-get install libpq-dev postgresql
# >>> sudo -u postgres -i  # (this will let you run commands as user 'postgres'. You should see username change in CLI)
# >>> createuser -s -P <username>  # add another superuser role to postresql. This will prompt to create a password
# NOTE: to avoid headaches involved with copying a database over, <username> should be the same as you use on your local machine
# >>> <ctrl+D>   # (quit sudo-ing as postgres)
# 
#
# Copying a postgreSQL database to EC2
# (on local computer) >>> pg_dump -C -h localhost -U <username> <db_name> > database.sql
# Copy this file to EC2 instance with scp command above. Then from EC2:
# >>> sudo adduser <username>  # add a linux user with same name as database owner
# >>> sudo -u <username> -i  # run commands as that user
# >>> psql < database.sql    # run SQL commands to copy database over
# This should copy entire database contents into your new EC2 database.
# You can delete the database.sql file.
#

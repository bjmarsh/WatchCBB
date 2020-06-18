#! /bin/bash

# you can run 'source ec2_setup.sh' in a fresh EC2 instance to run these commands
# and install the necessary packages. Note you need a requirements.txt file.

# make sure we've loaded most recent info from all package repositories
sudo apt-get update
# install system-wide required packages like python, gunicorn, nginx, postgresql
sudo apt-get install nginx gunicorn3 python3-pip python3-flask libpq-dev postgresql
# install project-specific python packages (requirements.txt lists all of your required python packages)
pip3 install -r requirements.txt


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
# NOTE: you have an option here. <username> will own the database. This can be 'ubuntu', 
#       but you will need to change the credentials that you use to connect in your code. 
#       Or, you can use the same username as you use on your local machine,
#       but you will need to create a new linux user with that name
# >>> createuser -s -P <username>  # add another superuser role to postresql. This will prompt to create a password
# >>> <ctrl+D>   # (quit sudo-ing as postgres)
# ONLY IF you choose to use the same username as on your local machine, create a new linux user:
# >>> sudo adduser <username>
# >>> sudo -u <username> -i  # start running commands as this new user
# Now do this, whether <username> is ubuntu or your new one
# >>> createdb <username>
#
# Copying a postgreSQL database to EC2
# (on local computer) >>> pg_dump -C -h localhost -U <username> --no-owner <db_name> > database.sql
# Copy this file to EC2 instance with scp command above. Then from EC2:
# ONLY IF you chose to own with same username as on local machine (i.e. not ubuntu):
# >>> sudo -u <username> -i  # run commands as that user
# Finally, copy database contents:
# >>> psql < database.sql    # run SQL commands to copy database over
# This should copy entire database contents into your new EC2 database.
# You can delete the database.sql file.
#
#
#
#############################################
# Some issues w/ tutorial and other problems
#############################################
#
# gunicorn.conf should actually be gunicorn.service. Will get errors otherwise
# 
# 
# 
# There is apparently already an nginx process running, which will overwrite
# the one you start (if you go to your webpage and you see an nginx splashscreen,
# this is what is happening). So you need to run:
#     sudo systemctl stop nginx
# and then
#     sudo systemctl start nginx
#
# 
# 
# By default apparently pip3 installs packages locally (i.e. in your user
# directory, not in the system-wide python packages location). Maybe this
# doesn't happen if you run 'sudo pip3', but if they are locally installed
# then the systemd gunicorn service won't work, because the package locations
# aren't in the PYTHONPATH by default.
#
# Now you don't really need to use systemd to start gunicorn, it is fine to
# run it in the background or in a screen session because the risk of random system
# reboot is probably very small. However, if you want to do this step, you have to
# modify it as follows:
#
# Make a wrapper script start_gunicorn.sh in your main project direcotory as follows:
# | #! /bin/bash
# | export PYTHONPATH=$PYTHONPATH:/home/<username>/.local/lib/python3.6/site-packages
# | gunicorn3 -b 0.0.0.0:8080 watchcbb.flask:app | tee log.txt
# 
# (need to modify <username> and the gunicorn command to whatever is specific to your app)
#
# Then in the gunicorn.service systemd file, the [Service] block should be as follows:
# | WorkingDirectory=/path/to/your/app/directory
# | ExecStart=/bin/bash start_gunicorn.sh
#
# This will run the wrapper instead of gunicorn directly, which makes sure to add the correct
# local directory to your PYTHONPATH
#
#
#
# XGBOOST issues
# xgboost requires cmake version >=3.16 to install correctly
# The default ubuntu version (that you get with sudo apt-get install cmake)
# is version 3.10, and so won't work.
#
# First, make sure there is no cmake already installed:
# >>> sudo apt-get remove cmake
# >>> pip3 uninstall cmake
#
# Now install the pip version of cmake:
# >>> pip3 install cmake
#
# If you run the below, you should see "cmake" among other things:
# >>> ls ~/.local/bin
#
# Another problem, pip installs things locally in ~/.local/bin, 
# but this isn't in your path, so the system doesn't know
# where to find it. Add it to your path like this:
# >>> export PATH=$PATH:~/.local/bin
#
# Now type 'cmake --version', you should see 3.17.
# If this is the case, 'pip3 install xgboost' should work.
#

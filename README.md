# WatchCBB

[![Build Status](https://travis-ci.org/bjmarsh/WatchCBB.svg?branch=master)](https://travis-ci.org/bjmarsh/WatchCBB)
[![Coverage Status](https://coveralls.io/repos/github/bjmarsh/WatchCBB/badge.svg?branch=master)](https://coveralls.io/github/bjmarsh/WatchCBB?branch=master)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/bjmarsh/WatchCBB/tree/master/notebooks/)
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bjmarsh/WatchCBB/master) -->

An app to provide personalized recommendations of entertaining college basketball games to watch,
built from scratch over 3 weeks as my Insight Data Science project.
Find it at [watchcbb.me](http://watchcbb.me).

<p align="center"><a href="http://watchcbb.me/">
<img src="./data/watchcbb_demo.gif" alt="WatchCBB demo" width="650"/>
</a></p>

## Contents

* The `data` and `notebooks` directories have their own READMEs with descriptions of the contents.
* `scripts` contains various utility scripts for scraping/cleaning data.
* `watchcbb` is a python module containing all of the "meat" of the app
  * `watchcbb.scrape`: submodule containing classes for scraping sports-reference/reddit/ESPN.
  * `watchcbb.flask`: submodule containing the flask web application
  * `watchcbb.utils`: utility functions for dealing with game data. Compile into season stats, add composite metrics, split into train/test by year, etc.
  * `watchcbb.efficiency`: function for computing SoS-adjusted efficiencies and other advanced stats
  * `watchcbb.sql`: utility class for interacting with PostgreSQL database. 
  * `watchcbb.reddit_utils`: utility functions for managing/parsing reddit gamethread data
  * `watchcbb.teams`: class/functions for handling team data
* `test` contains all of the unit testing framework
* `setup.sh`: add `watchcbb` to your python path
* `run_flask.py`: launch the flask web app contained in `watchcbb/flask`
* `ec2_setup.sh`: installation commands and many helpful comments for setting up an environment on an AWS EC2 instance.

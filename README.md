# WatchCBB

[![Build Status](https://travis-ci.org/bjmarsh/WatchCBB.svg?branch=master)](https://travis-ci.org/bjmarsh/WatchCBB)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/bjmarsh/WatchCBB/tree/master/notebooks/)
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bjmarsh/WatchCBB/master) -->

An app to provide personalized recommendations of entertaining college basketball games to watch.

<p align="center"><a href="http://watchcbb.me/">
<img src="./data/watchcbb_demo.gif" alt="WatchCBB demo" width="650"/>
</a></p>

## Contents

* The `data` and `notebooks` directories have their own READMEs with descriptions of the contents.
* `scripts` contains various utility scripts for scraping/cleaning data.
* `watchcbb` is a python module containing all of the "meat" of the app
  * `watchcbb.scrape`: submodule containing `SportsRefScrape` and `RedditCBBScrape` classes for scraping sports-reference/reddit.
  * `watchcbb.flask`: submodule containing the flask web application
  * `watchcbb.utils`: utility functions for dealing with game data. Compile into season stats, add composite metrics, split into train/test by year, etc.
  * `watchcbb.efficiency`: function for computing SoS-adjusted efficiencies and other advanced stats
  * `watchcbb.sql`: utility function for interacting with PostgreSQL database. It will automatically connect an sqlalchemy engine to the `cbb` database using credentials in `watchcbb/PSQL_CREDENTIALS.txt` when you import it.
  * `watchcbb.reddit_utils`: utility functions for managing/parsing reddit gamethread data
* `setup.sh`: add `watchcbb` to your python path
* `run_flask.py`: launch the flask web app contained in `watchcbb/flask`

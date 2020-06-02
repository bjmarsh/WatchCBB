import os
import datetime as dt

import numpy as np
import pandas as pd

from watchcbb.scrape.RedditCBBScrape import RedditCBBScrape

YEAR = 2018

startdate = dt.datetime(2017,11,10)
enddate = dt.datetime(2018,3,11)

lines = open('../watchcbb/scrape/REDDIT_CLIENT.txt').readlines()
client_id = lines[0].strip()
client_secret = lines[1].strip()
user_agent = 'testscript by /u/bboiler'

os.makedirs(f'../data/gamethreads/{YEAR}', exist_ok=True)

rcbb = RedditCBBScrape(client_id, client_secret, user_agent)

TIME_OFFSET = 5 # hours before UTC that we want to convert to

dfs = []
while startdate <= enddate:
    fname = f'../data/gamethreads/{YEAR}/{startdate.date()}.pkl.gz'
    if os.path.exists(fname):
        startdate += dt.timedelta(1)
        continue
        
    print("Getting game threads for date {0}".format(startdate.date()))
        
    df = rcbb.get_gamethreads_from_date(startdate)
    
    df.to_pickle(fname, compression='gzip')
    
    startdate += dt.timedelta(1)

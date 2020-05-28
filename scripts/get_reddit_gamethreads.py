import sys
sys.path.append("../scrape")
from RedditCBBScrape import RedditCBBScrape

import numpy as np
import pandas as pd


lines = open('../scrape/REDDIT_CLIENT.txt').readlines()
client_id = lines[0].strip()
client_secret = lines[1].strip()
user_agent = 'testscript by /u/bboiler'

os.makedirs('../data/gamethreads', exist_ok=True)

rcbb = RedditCBBScrape(client_id, client_secret, user_agent)

startdate = dt.datetime(2020,2,1)
enddate = dt.datetime(2020,2,29)

TIME_OFFSET = 5 # hours before UTC that we want to convert to

dfs = []
while startdate <= enddate:
    fname = '../data/gamethreads/{0}.pkl.gz'.format(startdate.date())
    if os.path.exists(fname):
        startdate += dt.timedelta(1)
        continue
        
    print("Getting game threads for date {0}".format(startdate.date()))
    
    start_epoch=int((startdate+dt.timedelta(hours=TIME_OFFSET)).timestamp())
    end_epoch=int((startdate+dt.timedelta(hours=TIME_OFFSET)+dt.timedelta(1)).timestamp())
    
    df = rcbb.get_gamethreads_from_date(startdate)
    
    df.to_pickle(fname, compression='gzip')
    
    startdate += dt.timedelta(1)

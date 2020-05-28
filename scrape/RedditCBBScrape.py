import os
import datetime as dt

import numpy as np
import pandas as pd

import praw
from psaw import PushshiftAPI


class RedditCBBScrape:
    def __init__(self, client_id, client_secret, user_agent):

        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                             user_agent=user_agent)
        self.api = PushshiftAPI(reddit)

    def get_gamethreads_from_date(self, date):
        if type(date) == dt.date:
            date = dt.datetime.combine(date, dt.time())
        if type(date) != dt.datetime:
            raise TypeError('Expecting datetime.datetime, got {0}'.format(type(date)))

        gen = self.api.search_submissions(title='"game thread"', subreddit='collegebasketball', 
                                          after=start_epoch, before=end_epoch, limit=500)
        return pd.DataFrame([x.__dict__ for x in gen])

if __name__=='__main__':
    lines = open('CLIENT.txt').readlines()
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

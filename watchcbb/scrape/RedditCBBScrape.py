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


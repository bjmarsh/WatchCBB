import os
import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd

import praw
from psaw import PushshiftAPI



class RedditCBBScrape:
    def __init__(self, client_id, client_secret, user_agent):

        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                             user_agent=user_agent)
        self.api = PushshiftAPI(self.reddit)

    def get_gamethreads_from_date(self, date):
        if type(date) == dt.date:
            date = dt.datetime.combine(date, dt.time())
        if type(date) != dt.datetime:
            raise TypeError('Expecting datetime.datetime, got {0}'.format(type(date)))

        gen = self.api.search_submissions(title='"game thread"', subreddit='collegebasketball', 
                                          after=start_epoch, before=end_epoch, limit=500)
        return pd.DataFrame([x.__dict__ for x in gen])

    def get_comments_from_post(self, post_id):
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        data = defaultdict(list)
        for comment in submission.comments:
            data['post_id'].append(post_id)
            data['comment_id'].append(comment.id)
            data['author'].append(comment.author)
            if comment.author_flair_text is None:
                data['author_flair'].append([])
            else:
                data['author_flair'].append([x.strip() for x in comment.author_flair_text.split('/')])
            data['text'].append(comment.body)
        df = pd.DataFrame(data, columns=['post_id', 'comment_id', 'author', 'author_flair', 'text'])

        return df

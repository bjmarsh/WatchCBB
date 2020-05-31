import os
import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

from watchcbb.scrape.RedditCBBScrape import RedditCBBScrape

YEAR = 2019

lines = open('../watchcbb/scrape/REDDIT_CLIENT.txt').readlines()
client_id = lines[0].strip()
client_secret = lines[1].strip()
user_agent = 'testscript by /u/bboiler'

os.makedirs(f'../data/gamethread_comments/{YEAR}', exist_ok=True)

rcbb = RedditCBBScrape(client_id, client_secret, user_agent)

dfs = []
for fname in glob.glob(f'../data/gamethreads/{YEAR}/*.pkl.gz'):
    with open(fname, 'rb') as fid:
        df = pd.read_pickle(fid, compression='gzip')
    dfs.append(df)
gt_df = pd.concat(dfs)
gt_df.sort_values('created', inplace=True)
gt_df.reset_index()

dfs = []
for post_id in tqdm(gt_df.id.values):
    dfs.append(rcbb.get_comments_from_post(post_id))
df = pd.concat(dfs)

df.to_pickle(f'../data/gamethread_comments/{YEAR}/comments.pkl.gz', compression='gzip')


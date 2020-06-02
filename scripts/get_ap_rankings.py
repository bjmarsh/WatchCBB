import os
import datetime as dt

import numpy as np
import pandas as pd

from watchcbb.scrape.SportsRefScrape import SportsRefScrape

sr = SportsRefScrape()
all_polls = {}
for season in range(2011,2021):
# for season in range(2020,2021):
    print("Getting AP polls for year "+str(season))
    polls = sr.get_ap_rankings(season)
    all_polls.update(polls)

dates = sorted(all_polls.keys())
ranks = [all_polls[d] for d in dates]

df = pd.DataFrame(ranks, index=dates, columns=['r'+str(i) for i in range(1,26)])
df.index.name = "date"

print(df.head())

with open('../data/ap_rankings.csv', 'w') as fid:
    df.to_csv(fid)

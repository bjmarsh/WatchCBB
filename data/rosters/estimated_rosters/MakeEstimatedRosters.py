from collections import defaultdict

import pandas as pd

year = 2021

df = pd.read_pickle("../{0}.pkl.gz".format(year-1), compression='gzip')
df.set_index('team_id', inplace=True)
old_rosters = df.to_dict(orient='index')
seniors = [l.strip() for l in open("seniors_{0}.txt".format(year-1)).readlines()]
nba = [l.strip() for l in open("nba_{0}.txt".format(year-1)).readlines()]
new_rosters = defaultdict(list)
for tname in sorted(old_rosters.keys()):
    players = old_rosters[tname]['players']
    new_rosters['team_id'].append(tname)
    new_rosters['players'].append([])
    new_rosters['WS'].append([])
    new_rosters['MP'].append([])
    for p in players:
        if p in seniors:
            continue
        if p in nba:
            continue
        new_rosters['players'][-1].append(p)
        new_rosters['WS'][-1].append(0.0)
        new_rosters['MP'][-1].append(0.0)

df = pd.DataFrame(new_rosters)
df.to_pickle('{0}.pkl.gz'.format(year), compression='gzip')


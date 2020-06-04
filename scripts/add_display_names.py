import json
import pandas as pd

d = json.load(open('display_names.json'))

df = pd.read_csv('../data/teams.csv')
dns = []
for tid in df.team_id:
    dns.append(d[tid])
df.insert(1, "display_name", pd.Series(dns))
print(df.head())

with open('../data/teams.csv', 'w') as fid:
    df.to_csv(fid, index=False)

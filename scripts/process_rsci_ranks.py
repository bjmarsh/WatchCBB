from collections import defaultdict
import pandas as pd

df_teams = pd.read_csv("../data/teams.csv")
dn2id = dict(zip(df_teams.display_name, df_teams.team_id))

translate = {
    "Miami" : "Miami FL",
    "Miami-FL" : "Miami FL",
    "Miami (FL)" : "Miami FL",
    "NC St." : "N.C. State",
    "St. Louis" : "Saint Louis",
    "Virgina" : "Virginia",
    "Central Florida": "UCF",
    "Cincinatti": "Cincinnati",
    "Florida International": "FIU",
    "St. Joseph's": "Saint Joseph's",
}

for year in range(2020,2009,-1):
# for year in range(2020,2019,-1):
    fin = open(f"../data/recruit_ranks/rsci_raw/{year}.csv")
    
    line = ""
    while not "RSCI," in line:
        line = fin.readline()

    idx = line.strip().split(',').index("RSCI")

    teams = defaultdict(int)

    for line in fin:
        sp = line.strip().split(',')
        r = int(sp[idx])
        dn = sp[-1]
        
        if dn in ["", "NBA G League", "Pro", "NBA", "G-League", "class of 2016", "Europe", "Juco", "prep school"]:
            continue

        dn = dn.replace("State","St.")
        dn = translate.get(dn, dn)
        if dn not in dn2id:
            raise Exception("Didn't find team "+dn)
        tid = dn2id[dn]

        teams[tid] += 101-r

    with open(f"../data/recruit_ranks/rsci_{year}.csv", 'w') as fid:
        fid.write("Rank,points,team_id\n")
        prev = -1
        prev_rank = 0
        for i,(tid,pts) in enumerate(sorted(teams.items(), key=lambda x:x[1], reverse=True)):
            if pts == prev:
                rank = prev_rank
            else:
                rank = i+1
            prev_rank = rank
            prev = pts
            fid.write(f"{rank},{pts},{tid}\n")

    fin.close()


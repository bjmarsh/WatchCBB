import os

from watchcbb.scrape.SportsRefScrape import SportsRefScrape


os.makedirs("../data/rosters", exist_ok=True)

sr = SportsRefScrape()
# for year in range(2020,2010,-1):
for year in range(2021,2020,-1):
    print(f"Getting rosters for {year}")
    fout = f"../data/rosters/{year}.pkl.gz"

    if year==2021:
        sr.get_roster_info(year, use_adv=False, est_file='../data/rosters/estimated_rosters/2021.pkl.gz', fout=fout)
    else:
        sr.get_roster_info(year, fout=fout)

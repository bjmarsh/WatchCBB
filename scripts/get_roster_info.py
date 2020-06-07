import os

from watchcbb.scrape.SportsRefScrape import SportsRefScrape


os.makedirs("../data/rosters", exist_ok=True)

sr = SportsRefScrape()
for year in range(2020,2010,-1):
    print(f"Getting rosters for {year}")
    fout = f"../data/rosters/{year}.pkl.gz"

    sr.get_roster_info(year, fout=fout)

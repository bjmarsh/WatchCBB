import os

from watchcbb.scrape.ESPNScrape import ESPNScrape


os.makedirs("../data/recruit_ranks", exist_ok=True)

sr = ESPNScrape()
for year in range(2020,2009,-1):
    print(f"Getting recruit rankings for {year}")
    fout = f"../data/recruit_ranks/espn_{year}.csv"

    sr.get_recruit_ranks(year, fout=fout)


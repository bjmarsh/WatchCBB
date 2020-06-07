import os

from watchcbb.scrape.SportsRefScrape import SportsRefScrape


os.makedirs("../data/game_data", exist_ok=True)

sr = SportsRefScrape()
for year in range(2020,2010,-1):
    fout = f"../data/game_data/game_data_{year}.csv"
    if os.path.exists(fout):
        continue

    print("Getting games for year "+str(year))
    sr.get_game_data(year, fout=fout, overwrite=True, verbose=True)

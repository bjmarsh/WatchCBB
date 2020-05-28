import os

from watchcbb.scrape.SportsRefScrape import SportsRefScrape


os.makedirs("../data/game_data", exist_ok=True)

sr = SportsRefScrape()
for year in range(2020,2019,-1):
    print("Getting games for year "+str(year))
    sr.get_game_data(year, fout=f"../data/game_data/game_data_{year}.csv", overwrite=True, verbose=True)

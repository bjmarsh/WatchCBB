import os
import datetime as dt

from watchcbb.scrape.SportsRefScrape import SportsRefScrape
from watchcbb.sql import SQLEngine

sql = SQLEngine('cbb')
srscrape = SportsRefScrape()

TODAY = dt.date(2020,2,17)
LOOKBACK = 2
OVERWRITE = True

SEASON = TODAY.year if TODAY.month < 6 else TODAY.year+1

for iprev in range(LOOKBACK, 0, -1):
    date = TODAY - dt.timedelta(iprev)
    print(f"Checking games for {date}")

    df_today = sql.df_from_query(""" 
        SELECT * FROM game_data
        WHERE "Season"={season} AND "Date"='{date}'
    """.format(season=SEASON, date=date))

    if df_today.shape[0] != 0:
        if OVERWRITE:
            print("Found existing games, deleting")
            sql.drop_rows('game_data', f""" "Date"='{date}' """)
        else:
            print("Found existing games, skipping date")
            continue

    print("Downloading game data")
    gids = srscrape.get_gids_on_date(date)
    df_newgames = srscrape.get_game_data(SEASON, gids=gids, verbose=True)

    print(df_newgames)

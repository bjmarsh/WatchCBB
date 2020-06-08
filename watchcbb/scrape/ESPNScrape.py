import os
import datetime as dt
import gzip

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from watchcbb.scrape.common import get_html


class ESPNScrape:
    """Class to perform various web-scraping routines from espn.com"""

    def __init__(self):
        pass

    
    def get_recruit_ranks(self, year, fout=None, 
                          teams_csv=os.path.join(os.path.dirname(__file__), '../../data/teams.csv')):
        """ """

        teams = []
        if year >= 2013:
            url = "http://insider.espn.com/college-sports/basketball/recruiting/classrankings?class={0}".format(year)
            html = get_html(url)
            soup = BeautifulSoup(html, "html.parser")
            
            for li in soup.find_all("li",{"class":"teamlist"}):
                teams.append(li.find("a").string)
                
        else:
            url = "http://insider.espn.com/college-sports/basketball/recruiting/archive/classrankings?classyear={0}".format(year)
            html = get_html(url)
            soup = BeautifulSoup(html, "html.parser")
            ul = soup.find("ul",{"class":"navlist"})
            for li in ul.find_all("li"):
                teams.append(li.find("p").string)
            url = "http://insider.espn.com/college-sports/basketball/recruiting/archive/classrankings?classyear={0}&viewmore=yes".format(year)
            html = get_html(url)
            soup = BeautifulSoup(html, "html.parser")
            ul = soup.find("ul",{"class":"navlist"})
            for li in ul.find_all("li"):
                teams.append(li.find("p").string)
                
        translate = {"Miami": "Miami FL",
                     "Ole Miss": "Mississippi",
                     "NC St.": "N.C. State",
                     "UConn": "Connecticut",
                     "Ucla": "UCLA",
        }
        for i in range(len(teams)):
            teams[i] = teams[i].replace(";","").replace(" State"," St.")
            if teams[i] in translate.keys():
                teams[i] = translate[teams[i]]

        if fout:
            df_teams = pd.read_csv(teams_csv)
            dn2id = dict(zip(df_teams.display_name, df_teams.team_id))
            fout = open(fout,'w')
            fout.write("Rank,team_id,display_name\n")
            for i,team in enumerate(teams):
                if team in dn2id.keys():
                    tid = dn2id[team]
                else:
                    raise Exception("Could not find team display name "+team)
                fout.write("{0},{1},{2}\n".format(i+1,tid,team,))
            fout.close()

        return teams


if __name__=="__main__":
    scrape = ESPNScrape()

    scrape.get_recruit_ranks(2012, fout="test.csv")

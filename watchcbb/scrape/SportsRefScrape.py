import datetime as dt
import gzip

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from watchcbb.scrape.common import get_html


class SportsRefScrape:
    """Class to perform various web-scraping routines from sports-reference.com/cbb"""

    def __init__(self):
        pass

    def get_gid(self, date, t1, t2):
        """ Return unique game id, with date and alphabetized teams, like 2020-02-15_indiana_purdue" """
        tnames = sorted([t1, t2])
        return "{0}_{1}_{2}".format(date,tnames[0],tnames[1])


    def get_team_list(self, season=2020):
        """ Return a list of all teams in D-I for a given season """

        teams_url = f"http://www.sports-reference.com/cbb/seasons/{season}-school-stats.html"
        teams_html = get_html(teams_url)
        teams_soup = BeautifulSoup(teams_html, "html.parser")
        teams = []
        table = teams_soup.find("table", id="basic_school_stats").find("tbody")
        for td in table.find_all("td", {"data-stat":"school_name"}):
            team = td.find("a")["href"].split("/")[3]
            teams.append(team)

        return teams


    def get_game_data(self, season, fout=None, overwrite=False, gids=None, teams=None, startdate=None, enddate=None, verbose=False):
        """Retrieve individual game statistics for a set of teams in a given season
        
        Parameters:
        season: year of the season (i.e. 2020 for 2019-20 season)
        fout: file to write output CSV to (None to not write to file)
        overwrite: True to overwrite file, False to append to it (taking care to avoid duplicates)
        gids: optional list of gids to get. If not None, this overrides anything in teams, startdate, enddate
        teams: list of team IDs (from sports-reference) to retrive games for.
               If None, use all teams in D-I for the given season
        startdate: date to start retrieving games, defaults to beginning of season
        enddate: date to end retrieving games, defaults to full season
        verbose: print extra info

        Returns: list of comma-separated strings, as would be written into the lines of a CSV
        """

        if teams is not None:
            if gids is not None:
                raise Exception("Only one of gids, teams can be non-null")
        else:
            if gids is None:
                teams = self.get_team_list(season)

        gids_to_get = None
        if gids is not None:
            gids_to_get = gids
            teams = [gid.split("_")[1] for gid in gids]
            teams = list(set(teams))
            
        gids = {}
        lines = {}
        rows = []

        # if we want to update the game file, record everything in the old file
        if fout is not None and overwrite==False:
            for line in open(fout).readlines()[1:]:
                sp = line.strip().split(",")
                date = sp[1]
                gid = self.get_gid(date,sp[3], sp[5])
                if date not in gids.keys():
                    gids[date] = []
                    lines[date] = []
                lines[date].append(line)
                gids[date].append(gid)

        stats = ["pts","fg","fga","fg3","fg3a","ft","fta","orb","trb","ast","stl","blk","tov","pf"]
        for team in teams:
            if verbose:
                print("Getting games for "+team+"...")

            url = f"http://www.sports-reference.com/cbb/schools/{team}/{season}-gamelogs.html"
            html = get_html(url)
            soup = BeautifulSoup(html, "html.parser")

            # this page only for "game type" (reg season, conf tourney, etc.) If before March, guaranteed Reg Season
            if enddate==None or enddate.month >= 2:
                url2 = "http://www.sports-reference.com/cbb/schools/{0}/{1}-schedule.html".format(team,season)
                html2 = get_html(url2)
                soup2 = BeautifulSoup(html2, "html.parser")

            table = soup.find("table", id="sgl-basic").find("tbody")
            for tr in table.find_all("tr"):
                if tr.get("id") == None:
                    continue

                date = tr.find("td", {"data-stat":"date_game"})
                if date.find("a") != None:
                    date = date.find("a").string
                else:
                    continue
                opp = tr.find("td", {"data-stat":"opp_id"})

                if startdate!=None and startdate > dt.date(*[int(x) for x in date.split("-")]):
                    continue 

                if enddate!=None and enddate < dt.date(*[int(x) for x in date.split("-")]):
                    continue 

                if opp.find("a")==None:
                    continue
                opp = opp.find("a")["href"].split("/")[3]
                gid = self.get_gid(date, team, opp)

                if gids_to_get is not None and gid not in gids_to_get:
                    continue

                datem1day = str(dt.date(*[int(x) for x in date.split("-")]) - dt.timedelta(1))
                gidm1day = self.get_gid(datem1day, team, opp)
                if date not in gids.keys():
                    gids[date] = []
                    lines[date] = []                
                if gid in gids[date] or (datem1day in gids.keys() and gidm1day in gids[datem1day]):
                    continue
                else:
                    gids[date].append(gid)

                if enddate==None or enddate.month >= 2:
                    gtype = soup2.find("td",{"csk":date}).find_parent("tr").find("td",{"data-stat":"game_type"}).string
                else:
                    gtype = "REG"
                if gtype == "REG":
                    gtype = "RG"
                if gtype == "CTOURN":
                    gtype = "CT"

                loc = tr.find("td", {"data-stat":"game_location"}).string
                if loc==None:    loc="H"
                elif loc=="@": loc="A"
                elif loc=="N": loc="N"
                else:
                    raise Exception(loc)

                numot = tr.find("td", {"data-stat":"game_result"})
                if numot.find("small") != None:
                    numot = int(numot.find("small").string.split("(")[1].split()[0])
                else:
                    numot = 0

                statdict = {}
                opp_statdict = {}
                getint = lambda x: (0 if x is None else int(x))
                for stat in stats:
                    statdict[stat] = getint(tr.find("td",{"data-stat":stat}).string)
                    opp_statdict[stat] = getint(tr.find("td",{"data-stat":"opp_"+stat}).string)

                if statdict["pts"] > opp_statdict["pts"]:
                    wd, ld = statdict, opp_statdict
                    wteam, lteam = team, opp
                else:
                    wd, ld = opp_statdict, statdict
                    wteam, lteam = opp, team
                    if loc=="H":   loc="A"
                    elif loc=="A": loc="H"

                rowvals = [season,date,gtype,wteam,wd["pts"],lteam,ld["pts"],loc,numot,
                           wd["fg"],wd["fga"],wd["fg3"],wd["fg3a"],wd["ft"],wd["fta"],wd["orb"],
                           wd["trb"]-wd["orb"],wd["ast"],wd["tov"],wd["stl"],wd["blk"],wd["pf"],
                           ld["fg"],ld["fga"],ld["fg3"],ld["fg3a"],ld["ft"],ld["fta"],ld["orb"],
                           ld["trb"]-ld["orb"],ld["ast"],ld["tov"],ld["stl"],ld["blk"],ld["pf"]
                ]
                rows.append(rowvals)

                string = ",".join([str(x) for x in rowvals]) + '\n'

                lines[date].append(string)

        colnames = ["Season","Date","Type","WTeamID","WScore","LTeamID","LScore","WLoc","NumOT",
                    "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl",
                    "WBlk","WPF","LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst",
                    "LTO","LStl","LBlk","LPF"
        ]
        if fout:
            fout = open(fout, 'w')
            fout.write(",".join(colnames)+'\n')
            for date in sorted(gids.keys()):
                for s in lines[date]:
                    fout.write(s)
            fout.close()

        return pd.DataFrame(rows, columns=colnames)


    def get_gids_on_date(self, startdate, enddate=None):
        """
        Return gids of all games between startdate and enddate (inclusive)
        If enddate is None, use only startdate
        """

        if enddate is None:
            enddate = startdate

        gids = []
        date = startdate
        while date <= enddate:
            url = f'https://www.sports-reference.com/cbb/boxscores/index.cgi?month={date.month:02d}&day={date.day:02d}&year={date.year}'
            html = str(get_html(url))
            if html.find("No games found") > -1:
                date += dt.timedelta(1)
                continue

            soup = BeautifulSoup(html, 'html.parser')
            for table in soup.find_all('table', {'class':'teams'}):
                td = table.find_all("tr")[0].find("td")
                a = td.find("a")
                if a==None:  # usually a non-DI team
                    continue
                if not a.has_attr("href"):
                    continue
                t1 = td.find("a")["href"].split("/")[3]
                td = table.find_all("tr")[1].find("td")
                a = td.find("a")
                if a==None:  # usually a non-DI team
                    continue
                if not a.has_attr("href"):
                    continue
                t2 = td.find("a")["href"].split("/")[3]
                
                gids.append(self.get_gid(date, t1, t2))

            date += dt.timedelta(1)

        return gids


    def get_ap_rankings(self, season):
        """
        Given the season, return a dictionary where the keys are dates 
        and values are length-25 lists giving the rankings 1-25 on that date.
        """

        url = f'https://www.sports-reference.com/cbb/seasons/{season}-polls.html'
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table', {'id':'ap-polls'})

        # get the poll dates
        polls = {}
        date_row = table.find('thead').find_all('tr')[2]
        for th in date_row.find_all('th')[2:]:
            s = th.string
            if s=="Pre":
                date = dt.date(season-1,10,1)
            elif s=="Final":
                date = dt.date(season,5,1)
            else:
                month = int(s.split('/')[0])
                day = int(s.split('/')[1])
                year = season
                if month > 7:
                    year -= 1
                date = dt.date(year,month,day)
            polls[date] = [[] for i in range(25)]

        sorted_dates = sorted(polls.keys())

        for tr in table.find('tbody').find_all('tr'):
            tds = tr.find_all('td')
            if len(tds)==0:
                continue
            tid = tr.find('th').find('a').get('href').split('/')[3]
            for date, td in zip(sorted_dates,tds[1:]):
                if td.string is not None and td.string != "":
                    idx = int(td.string)-1
                    polls[date][idx].append(tid)
                    
        for date in polls:
            if sum(len(teams) for teams in polls[date]) < 25:
                raise Exception(f'Less than 25 teams for date {date}')

        return polls

    def get_roster_info(self, season, teams=None, stats=["MP","WS"], fout=None, out_type="df"):
        """Get player IDs and statistics for a given season for every team in teams"""

        if teams==None:
            teams = self.get_team_list(season)

        data = {'team_id':[], 'players':[]}
        for stat in stats:
            data[stat] = []

        for tid in teams:
            print(f"Getting roster for {tid}")

            url = f"https://www.sports-reference.com/cbb/schools/{tid}/{season}.html"
            html = str(get_html(url))

            tablestartAdv = html.find('<table class="sortable stats_table" id="advanced"')
            tableendAdv = html.find("</table>",tablestartAdv)
            htmlAdv = html[tablestartAdv:tableendAdv+8]
            soupAdv = BeautifulSoup(htmlAdv, "html.parser")
            tableAdv = soupAdv.find("table", {"id":"advanced"})

            if tableAdv is None:
                print(f"WARNING: team {tid} not found for year {season}")
                continue
                
            data['team_id'].append(tid)
            for s in ['players']+stats:
                data[s].append([])
                
            for tr in tableAdv.find("tbody").find_all("tr"):
                data['players'][-1].append(tr.find("td",{"data-stat":"player"})["data-append-csv"])
                
                for stat in stats:
                    x = tr.find("td",{"data-stat":stat.lower()}).string
                    data[stat][-1].append(float(x) if x is not None else 0.0)

        if fout:
            if out_type == "df":
                df = pd.DataFrame(data, columns=['team_id','players']+stats)
                df.to_pickle(fout, compression='gzip')

        return data


if __name__=="__main__":
    
    sr = SportsRefScrape()

    # sr.get_game_data(2020, fout="../scratch/test.csv", overwrite=True, teams=['purdue'], verbose=True)

    # sr.get_roster_info(2017, teams=['utah-state'], fout='test.pkl.gz')

    # for gid in sr.get_gids_on_date(dt.date(2020,2,15), dt.date(2020,2,16)):
    #     print(gid)
    
    gids = sr.get_gids_on_date(dt.date(2020,2,16), dt.date(2020,2,16))
    print(sr.get_game_data(2020, gids=gids, verbose=True))

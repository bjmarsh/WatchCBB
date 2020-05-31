from collections import defaultdict

import urllib3
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

def get_html(url):
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    data = r.data
    r.release_conn()
    return data

url = "https://www.sports-reference.com/cbb/schools/"
html = get_html(url)
soup = BeautifulSoup(html, 'html.parser')

schools = []
table = soup.find('table', {'id':'schools'}).find('tbody')
for tr in table.find_all('tr'):
    td = tr.find('td')
    if td is None:
        continue
    name = td.find('a').get('href').split('/')[3]
    year_start = int(tr.find_all('td')[2].string)
    year_end = int(tr.find_all('td')[3].string)
    if year_end < 2000:
        continue

    schools.append((name,year_start,year_end))

data = defaultdict(list)
for tname, year_start, year_end in schools:
    print(f"Gettind data for {tname}...")

    url = f"https://www.sports-reference.com/cbb/schools/{tname}/"
    html = get_html(url)
    soup = BeautifulSoup(html, 'html.parser')

    loc = None
    conf = None    

    ps = soup.find_all('p')
    for p in ps:
        strong = p.find('strong')
        if strong is None:
            continue
        if strong.string == "Location:":
            loc = str(p).split('</strong>')[1].split("</p")[0].strip()
        if strong.string == "Conferences:":
            conf = p.find('a').string

    data["team_id"].append(tname)
    data["year_start"].append(year_start)
    data["year_end"].append(year_end)
    data["location"].append(loc)
    data["conference"].append(conf)

df = pd.DataFrame(data, columns=["team_id","conference","location","year_start","year_end"])
with open("../data/teams.csv", 'w') as fid:
    df.to_csv(fid, index=False)

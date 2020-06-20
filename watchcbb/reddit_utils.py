import os
import datetime as dt
import json
import re

import numpy as np
import pandas as pd


_TEAM_IDS = pd.read_csv(os.path.join(os.path.dirname(__file__),'../data/teams.csv')).team_id.values.tolist()
with open(os.path.join(os.path.dirname(__file__),'../data/name_substitutions.json')) as fid:
    _REPLACE_NAMES = json.load(fid)
_BAD = []

def date_from_timestamp(ts):
    """Convert UNIX timestamp to date of game.
       Timestamp is in UTC time. If before 10am, assign to previous day
    """
    datetime = dt.datetime.fromtimestamp(ts)
    date = datetime.date()
    if datetime.hour < 10:
        date -= dt.timedelta(1)
    return date

def parse_title(s):
    """Take the raw title of the (post)game thread and return a 2-tuple of team names.
       This strips away anything in parentheses/brackets, the score, rankings, and
       other unimportant info.
       Team names are lower-cased and spaces are replaced with hyphens.
    """
    s = s.lower()
    s = re.sub(r'\(.*\)', ' ', s)  # remove parenthetical expressions
    s = re.sub(r'\[.*\]', ' ', s)  # remove bracketed expressions
    s = re.sub(r'#[0-9]*', ' ', s) # remove rankings
    s = re.sub(r'[0-9]*-[0-9]*', ' ', s)  # remove scores
    s = s.split(',')[0]  # remove clauses after comma
    s = s.split(' in ')[0]  # remove things like " in 2OT"
    s = s.replace("'", '')  # remove aposrophes
    s = s.strip()
    s = " ".join(s.split())  # replace any multiple spaces with single space

    # Game threads are usually like "team1 @ team2", 
    # post-game threads usually like "team1 defeats team2"
    # Split here on a few select phrases
    for sp in ['@', 'defeats', 'has defeated', 'beats', 'defeat', 'vs.']:
        if sp in s:
            s = [x.strip() for x in s.split(sp)]
            break
    if type(s)==str:
        # From manual inspection this should be rare, and usually they aren't real game threads
#         raise Exception("Unexpected title format:"+s)
        return None

    # finally replace spaces with hyphens and return
    s = ['-'.join(x.split()) for x in s]
    return s



def fix_names(pair):
    """Replace names in each pair with their proper team_ids.
       If we can't guess how to do this, return None
    """
    if pair is None:
        return None
    if type(pair) not in [tuple, list]:
        return None
    ret = []
    for p in pair:
        if p in _TEAM_IDS:
            ret.append(p)
        elif p.replace("&","") in _TEAM_IDS:
            ret.append(p.replace("&",""))
        elif p.replace("st.","st") in _TEAM_IDS:
            ret.append(p.replace("st.","st"))
        elif p.replace("st.","saint") in _TEAM_IDS:
            ret.append(p.replace("st.","saint"))
        elif p.replace("uc","california") in _TEAM_IDS:
            ret.append(p.replace("uc","california"))
        elif p in _REPLACE_NAMES and _REPLACE_NAMES[p] in _TEAM_IDS:
            ret.append(_REPLACE_NAMES[p])
        else:
            _BAD.append(p)
            return None
    return ret

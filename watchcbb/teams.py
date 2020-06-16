
class Team:
    def __init__(self, team_id, display_name=None, flair_name=None, conference=None, location=None, 
                 year_start=None, year_end=None, color="#000000"):
        self.team_id = team_id
        self.display_name = self.team_id if display_name is None else display_name
        self.flair_name = flair_name
        self.conference = conference
        self.location = location
        self.year_start = year_start
        self.year_end = year_end
        self.color = color
        self.logo_url = f"https://d2p3bygnnzw9w3.cloudfront.net/req/202006091/tlogo/ncaa/{team_id}.png"
        
def teams_from_df(df):
    teams = {}
    for irow, row in df.iterrows():
        tid = row.team_id
        teams[tid] = Team(tid, row.display_name, row.flair_name, row.conference, row.location,
                          row.year_start, row.year_end, row.color)
    return teams

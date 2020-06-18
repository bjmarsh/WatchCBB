import os
import unittest

import pandas as pd

import watchcbb.teams

class TestTeams(unittest.TestCase):
    
    df = None

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/teams.csv"))

    def test_teams_from_df(self):
        teams = watchcbb.teams.teams_from_df(self.df)
        self.assertEqual(self.df.shape[0], len(teams))

if __name__=="__main__":
    unittest.main()

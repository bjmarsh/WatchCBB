import os
from copy import deepcopy
import unittest

import pandas as pd

import watchcbb.utils as utils
import watchcbb.efficiency as eff

class TestTeams(unittest.TestCase):
    
    df_games = None
    df_preseason = None

    @classmethod
    def setUpClass(cls):
        cls.df_games = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/test_game_data.csv'))
        utils.add_gid(cls.df_games)
        utils.add_poss(cls.df_games)
        cls.df_games['Lrank'] = -1
        cls.df_games['Wrank'] = -1
        cls.df_preseason = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/test_preseason_predictions.csv'))

    def test_compute_efficiency_ratings(self):
        tids = list(set(self.df_games.WTeamID) | set(self.df_games.LTeamID))
        tids.append('yale') # add a team that hasn't played yet

        ss_dict = utils.compute_season_stats(self.df_games, df_preseason=self.df_preseason,
                                             force_all_teams=True, tids=tids)
        self.assertRaises(Exception, eff.compute_efficiency_ratings, ss_dict)

        ss_df = utils.stats_dict_to_df(ss_dict)
        utils.add_advanced_stats(ss_df)
        ss_dict = utils.stats_df_to_dict(ss_df)
        # add dummy team to make sure it handles correctly
        ss_dict[2020]['blah'] = {'TFGA':0, 'rawpace':70, 'opps':[], 'HA':[]}

        copy = deepcopy(ss_dict)
        p = 0.9
        eff.compute_efficiency_ratings(copy, conv_param=0.9, preseason_blend=p)
        
        self.assertEqual(list(ss_dict.keys()), list(copy.keys()))
        for k in copy:
            self.assertEqual(list(ss_dict[k].keys()), list(copy[k].keys()))
        YEAR = 2020
        for tid in copy[YEAR]: 
            for sn in utils.ADVSTATNAMES:
                self.assertTrue("Tcorro"+sn in copy[YEAR][tid])
                self.assertTrue("Tcorrd"+sn in copy[YEAR][tid])
                self.assertTrue("Ocorro"+sn in copy[YEAR][tid])
                self.assertTrue("Ocorrd"+sn in copy[YEAR][tid])
            for s in ["pace", "CompositeRating", "CompositeOff", "CompositeDef", "CompositePace"]:
                self.assertTrue(s in copy[YEAR][tid])

            if copy[YEAR][tid]["TFGA"] > 0:
                x = p*copy[YEAR][tid]["preseason_eff"] + (1-p)*copy[YEAR][tid]["Tneteff"]
                self.assertTrue(abs(x-copy[YEAR][tid]["CompositeRating"]) < 1e-6)


if __name__=="__main__":
    unittest.main()

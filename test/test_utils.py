import os
import datetime as dt
import unittest
import pickle
import gzip
from itertools import permutations

import numpy as np
import pandas as pd

import watchcbb.utils as utils

class TestUtils(unittest.TestCase):

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

    def test_partition_games(self):
        first, second = utils.partition_games(self.df_games, frac=0.5)
        self.assertEqual(first, [0,2])
        self.assertEqual(second, [1,3,4])

    def test_compute_season_stats(self):
        ss_dict = utils.compute_season_stats(self.df_games, df_preseason=self.df_preseason)
        self.assertEqual(sorted(list(ss_dict.keys())), [2019,2020])
        self.assertEqual(sorted(list(ss_dict[2019].keys())), ['alabama-state', 'iowa-state', 'missouri'])
        self.assertEqual(ss_dict[2020]['purdue']['TScore'], 79+66)
        self.assertEqual(ss_dict[2020]['purdue']['opps'], ['green-bay','texas'])
        self.assertEqual(ss_dict[2020]['purdue']['HA'], ['H','H'])

        ss_dict = utils.compute_season_stats(self.df_games, force_all_teams=True, tids=['purdue'])

        self.assertRaises(Exception, utils.compute_season_stats, self.df_games, force_all_teams=True)
        

    def test_stats_dict_to_df(self):
        ss_dict = utils.compute_season_stats(self.df_games)
        ss_df = utils.stats_dict_to_df(ss_dict)
        self.assertEqual(set(ss_dict[2020]['purdue']), set(ss_df.columns)-set(['year','team_id']))

    def test_stats_df_to_dict(self):
        ss_dict = utils.compute_season_stats(self.df_games)
        ss_df = utils.stats_dict_to_df(ss_dict)
        ss_dict2 = utils.stats_df_to_dict(ss_df)
        self.assertEqual(ss_dict, ss_dict2)

    def test_add_advanced_stats(self):
        ss_dict = utils.compute_season_stats(self.df_games)
        ss_df = utils.stats_dict_to_df(ss_dict)
        utils.add_advanced_stats(ss_df)
        for sn in utils.ADVSTATFEATURES:
            self.assertTrue(sn in ss_df.columns)
        self.assertTrue('rawpace' in ss_df.columns)

    def test_training_workflow(self):
        """ test the functions compile_training_data, train_test_split_by_year, get_daily_predictions """
        
        LASTDATE = self.df_games.Date.values[-1]
        restricted_games = self.df_games[self.df_games.Date < LASTDATE]
        tids = list(set(self.df_games.WTeamID.values.tolist() + self.df_games.LTeamID.values.tolist()))
        ss_dict = utils.compute_season_stats(restricted_games, force_all_teams=True, tids=tids, df_preseason=self.df_preseason)
        ss_df = utils.stats_dict_to_df(ss_dict)
        utils.add_advanced_stats(ss_df)
        ss_dict = utils.stats_df_to_dict(ss_df)

        # add some dummy values
        for year in ss_dict:
            for tid in ss_dict[year]:
                d = ss_dict[year][tid]
                d['pace'] = 70.0
                d['Tneteff'] = np.random.rand()
                d['CompositeRating'] = np.random.rand()
                for sn in utils.ADVSTATNAMES:
                    d['Tcorro'+sn] = np.random.rand()
                    d['Tcorrd'+sn] = np.random.rand()

        os.makedirs('__tmp_season_stats')
        with gzip.open('__tmp_season_stats/{0}.pkl.gz'.format(LASTDATE), 'wb') as fid:
            pickle.dump((ss_dict, ss_df), fid, protocol=-1)

        # test compile_training_data
        data = utils.compile_training_data(restricted_games, ss_dict, sort='random')
        data = utils.compile_training_data(restricted_games, ss_dict, sort='alphabetical')
        self.assertRaises(Exception, utils.compile_training_data, restricted_games, ss_dict, sort='blah')

        # test train_test_split_by_year
        pca = utils.get_pca_model()
        train, test = utils.train_test_split_by_year(data, [2019], [2020], pca_model=pca)
        self.assertEqual(train.shape[0]+test.shape[0], data.shape[0])

        # test get_daily_predictions
        dates = [LASTDATE]
        preds = utils.get_daily_predictions(
            dates, self.df_games, 
            os.path.join(os.path.dirname(__file__),'models/test_models.pkl'), 
            '__tmp_season_stats', no_tqdm=True
        )

        self.assertEqual(preds.shape[0], 1)

        utils.get_daily_predictions(dates, self.df_games, 
                                    os.path.join(os.path.dirname(__file__),'models/test_models.pkl'), 
                                    'blah', no_tqdm=True)

        os.remove('__tmp_season_stats/{0}.pkl.gz'.format(LASTDATE))
        os.rmdir('__tmp_season_stats')
        
        

    def test_get_pca_model(self):
        pca = utils.get_pca_model()
        self.assertTrue(pca.fit is not None)
        self.assertTrue(pca.transform is not None)

    def test_is_upset(self):
        for r1,r2 in permutations([-1,1,5,10,15,20,25,30,40,100], 2):
            self.assertIsInstance(utils.is_upset(r1,r2), bool)

    def test_get_df_upset_prob(self):
        df = pd.DataFrame({
            'rank1': [-1, -1, 2, 10],
            'rank2': [-1, 1, 25, 10],
            'prob' : [0.5, 0.1, 0.8, 0.6],
            })
        up = df.apply(utils.get_df_upset_prob, axis=1).values
        self.assertTrue(np.amax(np.abs(up - [0, 0.1, 0.2, 0])) < 1e-6)

    def test_is_rivalry(self):
        df = pd.DataFrame({'gid': ['DATE_purdue_indiana', 'DATE_duke_oregon']})
        is_rivalry = df.apply(utils.is_rivalry, axis=1)
        self.assertTrue(is_rivalry[0])
        self.assertFalse(is_rivalry[1])

    def test_get_blend_param(self):
        self.assertTrue( 0 < utils.get_blend_param(0.314) < 1 )
        self.assertEqual(utils.get_blend_param(0), 1)
        self.assertEqual(utils.get_blend_param(1), 0)
        self.assertEqual(utils.get_blend_param(1.1), 0)

    def test_process_ap_sql(self):
        df_ap = pd.DataFrame(
            {'date': [dt.date(2020,11,1), dt.date(2020,11,8)],
             'r1': ['{purdue}', '{purdue, duke , pittsburgh}'],
             'r2': ['{kansas}', '{ syracuse, nc-state}'],
            }
        )
        utils.process_ap_sql(df_ap)
        self.assertEqual(df_ap.r1.values[0], ['purdue'])


if __name__=="__main__":
    unittest.main()

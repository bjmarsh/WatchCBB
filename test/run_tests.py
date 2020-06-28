#! /usr/bin/env python

import os
import unittest

if __name__=="__main__":

    test_dir = os.path.dirname(__file__)
    test_patt = "test_*.py"
    tests = unittest.TestLoader().discover(test_dir, test_patt)
    test_runner = unittest.TextTestRunner(verbosity=2)
    ret = test_runner.run(tests)

    if len(ret.errors)>0 or len(ret.failures)>0:
        exit(1)
    else:
        exit(0)

import unittest
import svgnest

if __name__ == '__main__':
    suits = []
    suits.append(unittest.TestLoader().loadTestsFromModule(svgnest))

    for suite in suits:
        unittest.TextTestRunner(verbosity=2).run(suite)

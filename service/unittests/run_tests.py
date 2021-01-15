import unittest
import pdf_validation
import ops_and_norms
import nomenclature


if __name__ == '__main__':
    suits = []
    for module in (pdf_validation, ops_and_norms, nomenclature):
        suite = unittest.TestLoader().loadTestsFromModule(module)
        unittest.TextTestRunner(verbosity=2).run(suite)



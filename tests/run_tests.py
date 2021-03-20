import unittest
import predict_price
# import pdf_validation
# import ops_and_norms
# import nomenclature


if __name__ == '__main__':
    suits = []
    for module in (predict_price,):
        suite = unittest.TestLoader().loadTestsFromModule(module)
        unittest.TextTestRunner(verbosity=2).run(suite)



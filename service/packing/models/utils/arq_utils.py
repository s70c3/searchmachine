from math import log

class Pandas:
    # Series transforms
    logscale = lambda v: log(v)
    empty = lambda v: v
    
    
class Numbers:
    @staticmethod
    def shortify(n):
        if n > 1e6:
            return str(round(n/1e6, 1)) + 'M'
        elif n > 1e4:
            return str(round(n/1e3)) + 'K'
        else:
            return n
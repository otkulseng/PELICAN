import numpy as np
import logging

import numpy as np
from itertools import zip_longest, chain


def interleave(elems):
    log = logging.getLogger('')
    log.warning("Using interleave.\
                By def, this only works \
                if the lists are approximately the same length")

    minlen = len(elems[0])
    for elem in elems:
        minlen = min(minlen, len(elem))

    res = np.array([x for x in chain(*zip_longest(*elems)) if x is not None])
    return res



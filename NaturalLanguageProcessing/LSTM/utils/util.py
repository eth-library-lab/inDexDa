import numpy as np
import math


def findMax(data):
    data = np.asarray(data)
    return np.amax(data)


def roundToNextFifty(x, base=50):
    return base * math.ceil(x / base)

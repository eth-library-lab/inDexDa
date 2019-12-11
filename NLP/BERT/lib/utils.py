import os
import sys


# Disable command line printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore command line printing
def enablePrint():
    sys.stdout = sys.__stdout__

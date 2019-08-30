import os
import ssl
import time
import json
import urllib.request
import utils.command_line as progress

from utils.url_util import get_url, check_url
from utils.json2mongodb import json2mongodb
from utils.webpage2dict import webpage2dict

LOG_FILE = 'log/log.txt'

def compile_database(options, OUTPUT_FILE):
    '''
    Reads through file containing online paper links, iterates through, and deposits
    information about the paper into a Mongo database.

    :params options: dictionary from argparser
            OUTPUT_FILE: path to file containing links
    :return N/A
    '''
    idx = last_entry(LOG_FILE)

    try:
        if os.path.exists(OUTPUT_FILE):
            length = doclength(OUTPUT_FILE)
            with open(OUTPUT_FILE, 'r') as f:
                for i, line in enumerate(f):
                    progress.printProgressBar(i + 1, length, prefix='Progress:',
                                          suffix='Complete', length=50)

                    if idx is not None and i < idx:
                        continue

                    info = webpage2dict(line)
                    json2mongodb(info)

    except ssl.CertificateError as e:
        print("Disconnected from Internet")
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w') as log:
                log.write(str(i) + '\n')
        else:
            os.mkdir(LOG_FILE)
            with open(LOG_FILE, 'w') as log:
                log.write(str(i) + '\n')


def doclength(fname):
    '''
    Gets number of lines within a file
    :param fname: path to file
    :return integer
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
        return i + 1


def last_entry(log_file):
    '''
    Opens log file and returns line number of last entry read.
    Catches non-existant files, empty files, and non-integer entries

    :param log_file: path to log file
    :return integer or None
    '''
    if os.path.exists(log_file):
        with open(log_file, 'r') as log:
            log_entry = log.read()
            if os.path.getsize(log_file) > 0:
                try:
                    return int(log_entry)
                except ValueError:
                    print('Log file does not contain an integer')
                    return None
            else:
                print('Log file does not contain an integer')
                return None
    else:
        return None

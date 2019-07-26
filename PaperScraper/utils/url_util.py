import urllib.request
from requests import get


def get_url(url):
    '''
    Checks that a given url is reachable and returns its contents
    :param url: A URL string
    :return: url contents or None
    '''
    try:
        urllib.request.urlopen(url)
        return get(url)
    except urllib.request.HTTPError:
        print("Website does not exist")
        return None
    except urllib.request.URLError:
        print("Website does not exist")
        return None


def check_url(url):
    '''
    Checks that a given url is reachable
    :param url: A URL string
    :return: True if site exists, False if otherwise
    '''
    try:
        urllib.request.urlopen(url)
        return True
    except urllib.request.HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        return False
    except urllib.request.URLError as e:
        print('Failed to reach a server.')
        print('Reason: ', e.reason)
        return False

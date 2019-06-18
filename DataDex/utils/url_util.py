from requests import get
from contextlib import closing
from requests.exceptions import RequestException


def get_url(url):
    '''
    Open url
    '''
    try:
        with get(url) as resp:
            print(resp)
            input("CHECK RESP")
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def is_good_response(resp):
    '''
    Check url response, returns True is working
    '''
    content_type = resp.headers['Content-Type'].lower()
    return(resp.status_code == 200 and content_type is not None and
           content_type.find('html') > -1)


def log_error(e):
    print(e)

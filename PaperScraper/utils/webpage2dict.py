import json
import time
import urllib.request
from bs4 import BeautifulSoup
from utils.url_util import check_url


def MergeDicts(dict_list):
    '''
    Merges a list of dictionaries into a single dictionary
    :param dict_list: A list of dictionaries
    :return: Dictionary
    '''
    result = {}
    for dictionary in dict_list:
        result.update(dictionary)
    return result


def webpage2dict(url):
    '''
    Reads webpage and stores specific information into json file
    :param url: A url link
    :return: json file
    '''
    while True:
        try:
            html = BeautifulSoup(urllib.request.urlopen(url), 'html.parser')

            content = html.findAll("div", {"id": "content"})
            content = content[1]

            # Get required information from site. Set to None is field does not exist
            try:
                title = content.find("h1", {"class": "title mathjax"}).text
            except AttributeError:
                title = None
                # print("No Title")
            try:
                authors = content.find("div", {"class": "authors"}).text
            except AttributeError:
                authors = None
                # print("No Authors")
            try:
                abstract = content.find("blockquote", {"class": "abstract mathjax"}).text
            except AttributeError:
                abstract = None
                # print("No Abstract")
            try:
                subject = content.find("span", {"class": "primary-subject"}).text
            except AttributeError:
                subject = None
                # print("No Subject")
            try:
                date = content.find("div", {"class": "dateline"}).text
                date = date.lstrip().replace("(Submitted on ", "").replace(")", "")
            except AttributeError:
                date = None
                # print("No Date")
            try:
                DOI = content.find("a", {"class": "link-https link-external"}).text
            except AttributeError:
                DOI = None
                # print("No DOI")

            # From info make dictionaries to place into json file
            if title is not None:
                dict_title = {title.split(':')[0]: title.split(':')[1]}
            else:
                dict_title = {"Title": ''}
            if authors is not None:
                dict_authors = {authors.split(':')[0]: authors.split(':')[1]}
            else:
                dict_authors = {"Authors": ''}
            if abstract is not None:
                dict_abstract = {abstract.split(':')[0]: abstract.split(':')[1].replace('\n', '')}
            else:
                dict_abstract = {"Abstract": ''}
            if date is not None:
                dict_date = {"Date": date}
            else:
                dict_date = {"Date": ''}
            if subject is not None:
                dict_subject = {"Subject": subject}
            else:
                dict_subject = {"Subject": ''}
            if DOI is not None:
                dict_DOI = {"DOI": DOI}
            else:
                dict_DOI = {"DOI": ''}

            paper = MergeDicts([dict_title, dict_authors, dict_abstract, dict_subject, dict_date,
                                dict_DOI])
            return paper

        except urllib.request.URLError as e:
            time.sleep(300)

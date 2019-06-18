# from urllib.parse import urljoin
# from bs4 import BeautifulSoup
# from selenium import webdriver
import datetime
import requests

from utils.url_util import get_url
from dateutil.relativedelta import relativedelta


def get_link(config):
    topic = Topic(config)
    print(topic.links)


class Topic:
    def __init__(self, config):
        """
        Create an for storing links to papers in a given topic

        :param config: config.json file containing information about which
                        topic papers will be found in
        """

        self.config = config
        self.topic = config['topic']
        self.archive_url = config['archive_html']
        self.page_links = self.get_page_links()

        self.download_pdfs()

    def get_dates(self):
        """
        Get dates in YYMM configuration between date when database started
            publishing the papers and current month and year
        """
        date_list = []
        date = []
        result = []

        start_month = int(self.config['start_month'])
        start_year = int(self.config['start_year'])
        end_month = int(self.config['end_month'])
        end_year = int(self.config['end_year'])

        end = datetime.date(end_year, end_month, 1)
        current = datetime.date(start_year, start_month, 1)

        while current <= end:
            date_list.append(current)
            current += relativedelta(months=1)

        for date in date_list:
            result.append(str(date.year % 100).zfill(2) +
                          str(date.month).zfill(2))

        return result

    def get_page_links(self):
        """
        Get links based on dates (YYMM) which point towards the archived
            webpage
        """
        links = []
        dates = self.get_dates()

        for date in dates:
            links.append(self.archive_url + date + '?skip=0&show=2000')

        return links

    def download_pdfs(self):
        # # for url in self.page_links:
        #     response = requests.get(url)
        #     with open('data/arXiv/test.pdf', 'wb') as f:
        #         f.write(response.content)
        #         print("First pdf printed")
        #         input("TEST PDF PRINTING")
        response = get_url('https://arxiv.org/pdf/1906.00001.pdf')
        # response = requests.get('https://arxiv.org/pdf/1906.00001.pdf')
        with open('data/arXiv/test.pdf', 'wb') as f:
            # f.write(response.content)
            f.write(response)
            print("First pdf printed")
            input("TEST PDF PRINTING")

# def arXiv(config):
#     '''
#     Get link to pdfs of papers. Unique for different topics
#     and fields.
#     '''

#     url = config['archive_html']
#     months = ["%.2d" % i for i in range(1, 13)]
#     years = []
#     paper_links = []

#     raw_html = get_url(url)
#     html = BeautifulSoup(raw_html, 'html.parser')
#     for li in html.find_all('li'):
#         if "Article statistics by year:" in li.text:
#             for a in li.find_all('a', href=True):
#                 years.append(a.text)
#         else:
#             continue
#     print(years)
#     input("WAIT")

#     wrong_dates = dates()

#     for year in years:
#         for month in months:
#             date = str(year) + str(month)
#             if date not in wrong_dates:
#                 paper_links.append(urljoin(config['html'], date))

#     print(len(paper_links))

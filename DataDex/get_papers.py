# from urllib.parse import urljoin
from bs4 import BeautifulSoup
# from selenium import webdriver
import datetime
import utils.url_util as urlutil
import urllib.request
# from utils.url_util import get_url, check_url
from dateutil.relativedelta import relativedelta


def get_link(config):
    topic = Topic(config)
    print(topic.pdf_links)


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

        self.pdf_links = self.search_archive()

        # self.download_pdfs()

    def get_dates(self):
        """
        Get dates in YYMM configuration between date when database started
            publishing the papers and current month and year
        :return result: list of eligible years and months that the database
                            has papers for (YYMM)
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
        :return links: link to archive pages for all years and months of the
                        database
        """
        links = []
        dates = self.get_dates()

        for date in dates:
            links.append(self.archive_url + date + '?skip=0&show=2000')

        return links

    def get_pdf_links(self, content):
        """
        Scrapes the given content of a webpage to extract all the pdf links

        :param content: portion of webpage contained within the div "content"
                         tag
        :return pdf_links: link to archive pages for all years and months of
                        the database
        """
        pdf_links = []
        for a in content.find_all(lambda tag: tag.name == "a" and "pdf"
                                  in tag.text):
            pdf_links.append('https://arxiv.org' + a.get('href'))
        return pdf_links

    def search_archive(self):
        """
        Scrapes the given content of a webpage to extract all the pdf links

        :return links: links to all the pdfs of a given year and month
        """
        next_page = []
        links = []

        for page in self.page_links:
            if urlutil.check_url(page):
                html = BeautifulSoup(urllib.request.urlopen(page),
                                     'html.parser')
                content = html.find("div", {"id": "content"})
                small_tags = []
                for small in content.find_all(lambda tag: tag.name ==
                                              "small" and "total of" in
                                              tag.text):
                    small_tags.append(small)

                for a in small_tags[0].find_all("a"):
                    next_page.append("https://arxiv.org" + a.get('href'))

                if not next_page:
                    links.extend(self.get_pdf_links(content))
                else:
                    for stack in next_page:
                        links.extent(self.get_pdf_links(content))

        return links

    def download_pdfs(self):
        """
        Opens pdfs as specified by links and saves them
        """
        # # for url in self.page_links:
        #     response = requests.get(url)
        #     with open('data/arXiv/test.pdf', 'wb') as f:
        #         f.write(response.content)
        #         print("First pdf printed")
        #         input("TEST PDF PRINTING")
        response = urlutil.get_url('https://arxiv.org/pdf/1906.00001.pdf')
        # response = get_url('http://thisdoesntexistforsurebababa.com')
        if response is not None:
            with open('data/arXiv/test.pdf', 'wb') as f:
                f.write(response.content)
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

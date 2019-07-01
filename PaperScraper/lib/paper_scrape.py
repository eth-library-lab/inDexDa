import datetime
import urllib.request
import utils.url_util as urlutil

from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta


class PaperScrape:
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
            print(page)
            if urlutil.check_url(page):
                html = BeautifulSoup(urllib.request.urlopen(page),
                                     'html.parser')
                content = html.find("div", {"id": "content"})
                small_tags = []

                # Find how many pages of pdfs the site has for specified date
                for small in content.find_all(lambda tag: tag.name ==
                                              "small" and "total of" in
                                              tag.text):
                    small_tags.append(small)

                # If site has no papers available for specified date
                if not small_tags:
                    continue

                # Compile list of all pages with pdfs
                for a in small_tags[0].find_all("a"):
                    next_page.append("https://arxiv.org" + a.get('href'))

                # For each page, get links for all pdfs
                if not next_page:
                    links.extend(self.get_pdf_links(content))
                else:
                    for stack in next_page:
                        links.extend(self.get_pdf_links(content))

        return links

    def download_pdfs(self):
        """
        Opens pdfs as specified by links and downloads them
        """

        for url in self.pdf_links:
            response = urlutil.get_url(url)
            if response is not None:
                with open('data/arXiv/test.pdf', 'wb') as f:
                    f.write(response.content)
                    print("First pdf printed")
                    input("TEST PDF PRINTING")
            else:
                print("Failed to download: {}".format(url))

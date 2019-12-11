import io
import os
import re
import json
import pdfminer
import requests

from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage


class ArXivScraper():
    def __init__(self, paper):
        # Perform layout analysis for all text
        self.laparams = pdfminer.layout.LAParams()
        setattr(self.laparams, 'all_texts', True)
        self.paper = paper

    def downloadPaper(self):
        '''
        Downloads the paper pdf into a temporary file

        :params  N/A
        :return  N/A
        '''
        pdf = requests.get(self.paper["Link"])
        with open('temp_pdf.pdf', 'wb') as f:
            f.write(pdf.content)

    def extractTextFromPdf(self):
        '''
        Uses PDFMiner objects to read the pdf temp file and extract the text from it

        :params  N/A
        :return  N/A
        '''
        file_handle = io.StringIO()
        resource_manager = PDFResourceManager()
        converter = TextConverter(resource_manager, file_handle, laparams=self.laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        # Read pdf into pdfminer object, then delete the temp file
        with open("temp_pdf.pdf", 'rb') as fh:
            for page in PDFPage.get_pages(fh,
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)
        os.remove('temp_pdf.pdf')

        self.text = file_handle.getvalue()

    def analyzeText(self):
        '''
        Finds name of dataset if included.
            - Scan for word dataset
            - Take preceeding words as long as there is no period and all previous words
                with capitalization are taken.
        Extracts any links in the body paragraphs.

        :params  N/A
        :return  N/A
        '''

        # Find dataset name in text
        sentences = self.text.replace('\n', ' ').split('.')

        dataset_name = []
        for sentence in sentences:
            words = sentence.split()
            if 'dataset' in words:
                idx = words.index('dataset')
                while True:
                    if words[idx - 1][0].isupper() and words[idx - 1] is not '.':
                        dataset_name.insert(0, words[idx - 1])
                        idx = idx - 1
                    else:
                        break

        if dataset_name != []:
            dataset_name.append('Dataset')
            dataset_name = ' '.join(dataset_name)
        else:
            dataset_name = []

        # Find all links in text
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', self.text)
        urls = self.extendUrls(urls)
        input(urls)

        self.dataset_name = dataset_name
        self.urls = urls

    def extendUrls(self, urls):
        '''
        Sometimes a new paragraph or space will cause regex to not identify the full
        url, so we check if the next character is a space or newline

        :params  url: string
        :return  extended_urls: string
        '''
        extended_urls = []
        for url in urls:
            url_start_idx = self.text.find(url)
            url_idx = self.text.find(url) + len(url)

            bad_chars = ['(', ')', '{', '}', '<', '>', ' ']

            while True:
                if self.text[url_idx + 1] not in bad_chars and not self.text[url_idx + 1].isupper():
                    url_idx += 1
                else:
                    break

            extended_urls.append(self.text[url_start_idx:url_idx].replace('\n', ''))

        return extended_urls


    def updatePaper(self):
        '''
        Updates paper dict with dataset name and possible links to dataset if they were
        found in the paper pdf.

        :params  N/A
        :return  N/A
        '''
        self.paper.update({'Dataset Name': self.dataset_name})
        self.paper.update({'Possible Dataset Links': self.urls})

    def saveResults(self):
        with open('temp_pdf.txt', 'w') as f:
            f.write(self.text)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(current_dir, '../../data/results.json')
    with open(datapath, 'r') as f:
        contents = f.read()
        papers = json.loads(contents)

    test = ArXivScraper(papers[0])
    test.downloadPaper()
    test.extractTextFromPdf()
    # test.saveResults()
    test.analyzeText()
    test.updatePaper()
    input(test.paper)

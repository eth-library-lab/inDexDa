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

from normalize_text import Normalize

class ArXivScraper():
    def __init__(self, config):
        '''
        Skim through the paper's text to try and find any urls within that could
        point towards a dataset and try to find the name of the dataset.

        :params  paper: dict of information about paper
        :return  paper: updated version of input
        '''
        self.laparams = pdfminer.layout.LAParams()
        setattr(self.laparams, 'all_texts', True)
        self.paper = config[0]

    def extract(self):
        '''
        Runs all scripts for the class

        :params  N/A
        :return  paper: updated dict of paper info
        '''
        self.downloadPaper()
        self.extractTextFromPdf()
        self.analyzeText()
        self.updatePaper()

        return self.paper

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

        # with open('temp_paper.txt', 'w') as f:
        #     f.write(self.text)

    def analyzeText(self):
        '''
        Calls functions to find the dataset name and possible urls to the dataset within
            the text.

        :params  N/A
        :return  N/A
        '''

        # Find dataset name
        sentences = self.text.replace('\n', ' ').split('.')
        dataset_names = self.findDatasetName(sentences)

        # Find all links in text
        urls = self.findUrls()

        self.dataset_names = dataset_names
        self.urls = urls

    def findDatasetName(self, sentences):
        '''
        Finds name of dataset if included.
            - Scan for word dataset
            - Take preceeding words as long as there is no period and all previous words
                with capitalization are taken.
        Extracts any links in the body paragraphs.

        :params  sentences: list of sentences from the text
        :return  names: list of possible dataset names
        '''
        useless_words = ['The', 'A', 'This', 'Other', 'Most', 'Some', 'All', 'These',
                         'Those', 'These', 'Our', 'Their', 'Several', 'Each', 'Which',
                         'Thus', 'As', 'Each', 'Suppose', 'Existing', '.']

        # Find instances of the word 'dataset' in the text
        dataset_names = []
        for sentence in sentences:
            words = sentence.split()

            if 'dataset' in words:
                dataset_name = []

                idx = words.index('dataset')
                while True:
                    if not idx > 0:
                        break
                    if words[idx - 1][0].isupper() and not any(words[idx - 1] == word for word in useless_words):
                        dataset_name.insert(0, words[idx - 1])
                        idx = idx - 1
                    else:
                        break
                if dataset_name != []:
                    dataset_names.append(dataset_name)

            if 'Dataset' in words:
                dataset_name = []

                idx = words.index('Dataset')
                while True:
                    if not idx > 0:
                        break
                    if words[idx - 1][0].isupper() and not any(words[idx - 1] == word for word in useless_words):
                        dataset_name.insert(0, words[idx - 1])
                        idx = idx - 1
                    else:
                        break
                if dataset_name != []:
                    dataset_names.append(dataset_name)

            if 'datasets' in words:
                dataset_name = []

                idx = words.index('datasets')
                while True:
                    if not idx > 0:
                        break
                    if words[idx - 1][0].isupper() and not any(words[idx - 1] == word for word in useless_words):
                        dataset_name.insert(0, words[idx - 1])
                        idx = idx - 1
                    else:
                        break
                if dataset_name != []:
                    dataset_names.append(dataset_name)

        # Append name to include 'Dataset' at the end
        names = []
        if dataset_names != []:
            for dataset in dataset_names:
                dataset.append('Dataset')
                dataset = ' '.join(dataset)
                names.append(dataset)

        # Remove duplicates
        names = list(set(names))
        return names

    def findUrls(self):
        '''
        Finds any urls within the text

        :params  N/A
        :return  new_urls: list of urls
        '''
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', self.text)
        urls = self.extendUrls(urls)

        # Remove non-ASCII characters
        new_urls = []
        for url in urls:
            # Remove non-ASCII characters
            url_new = Normalize(url, toLower=False, removePunctuation=False, removeNonASCII=True,
                                 removeContradictions=False, denoise=False, removeStopWords=False,
                                 stem=False, lemmatize=False, tokenize=False)

            tmp_list = []
            split_urls = url_new.normalized_text.split('http')
            split_urls.remove('')
            for url in split_urls:
                if url[0] == 's' or url[0] == ':':
                    url = "http" + url
                tmp_list.append(url)

                try:
                    url.replace(',', '')
                except:
                    continue

                try:
                    url.replace(' ', '')
                except:
                    continue


            new_urls.extend(tmp_list)

        new_list = tmp_list

        # Remove duplicates
        new_urls = list(set(new_urls))
        return new_urls

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

            bad_chars = ['(', ')', '{', '}', '<', '>', ' ', ',', '[', ']', '\\']

            while True:
                if url_idx + 1 < len(self.text):
                    if self.text[url_idx + 1] not in bad_chars and not self.text[url_idx + 1].isupper():
                        url_idx += 1
                    else:
                        break
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
        self.paper.update({'Dataset_Names': self.dataset_names})
        self.paper.update({'Possible_Dataset_Links': self.urls})

    def saveResults(self):
        with open('temp_pdf.txt', 'w') as f:
            f.write(self.text)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(current_dir, '../../data/results.json')
    with open(datapath, 'r') as f:
        contents = f.read()
        papers = json.loads(contents)

    newpapers = []
    for paper in papers:
        config = [paper, 'Random']
        scraper = ArXivScraper(config)
        newpaper = scraper.extract()

        newpapers.append(newpaper)

    with open('test_file.json', 'w') as f:
        json.dump(newpapers, f, indent=4)

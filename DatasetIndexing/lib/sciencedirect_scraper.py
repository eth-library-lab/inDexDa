import re
import os
import json
import time
import requests

from xml.dom import minidom
from xml.etree import ElementTree
from termcolor import colored


class ScienceDirectScraper():
    def __init__(self, config):
        '''

        :param config: namedtuple containing information about which
                        topic papers will be found in
        '''
        self.paper = config[0]
        self.apikey = config[1][1].apikey
        self.doi = self.paper['DOI']
        self.openaccess = 0

    def extract(self):
        '''
        Runs all scripts for the class

        :params  N/A
        :return  paper: updated dict of paper info
        '''
        self.extractTextFromXml()
        self.analyzeText()
        self.updatePaper()

        return self.paper

    def extractTextFromXml(self):
        '''
        Uses the Article Retrieval API to acquire full paper in xml form. Gets
        the body text from the paper.

        :params  N/A
        :return  N/A
        '''
        try:
            # Call API
            article = self.APIRequest()
        except Exception as error:
            input(error)
            raise(error)

        # If no article found, stop
        if article is None:
            self.text = None
            return

        with open('test.xml', 'wb') as f:
            mydata = ElementTree.tostring(article)
            f.write(mydata)

        # XML will only contain body text if it is deemed open-access. If it is
        #   not, return None.
        coredata = article.find('{http://www.elsevier.com/xml/svapi/article/dtd}coredata')
        openaccess = coredata.find('{http://www.elsevier.com/xml/svapi/article/dtd}openaccess')

        # If article is not open-access, we cannot scrape more information about it
        if not int(openaccess.text) or openaccess.text is None:
            self.openaccess = 0
            self.text = None
            return
        else:
            self.openaccess = 1

        try:
            # We parse the xml such that we can retrieve the relevant information about
            #   the body text of the paper
            paragraphs = []
            text = article.find('{http://www.elsevier.com/xml/svapi/article/dtd}originalText')
            doc = text.find('{http://www.elsevier.com/xml/xocs/dtd}doc')
            serial_item = doc.find('{http://www.elsevier.com/xml/xocs/dtd}serial-item')

            try:
                article_text = serial_item.find('{http://www.elsevier.com/xml/ja/dtd}article')
            except AttributeError:
                article_text = serial_item.find('{http://www.elsevier.com/xml/ja/dtd}converted-article')

            body = article_text.find('{http://www.elsevier.com/xml/ja/dtd}body')
            sections = body.find('{http://www.elsevier.com/xml/common/dtd}sections')

            for section in sections.findall('{http://www.elsevier.com/xml/common/dtd}section'):
                for child in section:
                    if '{http://www.elsevier.com/xml/common/dtd}para' in child.tag:
                        paragraphs.append(ElementTree.tostring(child, encoding="us-ascii", method="xml"))
                    if '{http://www.elsevier.com/xml/common/dtd}section' in child.tag:
                        for para in child.findall('{http://www.elsevier.com/xml/common/dtd}para'):
                            paragraphs.append(ElementTree.tostring(para, encoding="us-ascii", method="xml"))

            for list_item in article.iter('{http://www.elsevier.com/xml/common/dtd}list-item'):
                for item_text in list_item.findall('{http://www.elsevier.com/xml/common/dtd}para'):
                    paragraphs.append(ElementTree.tostring(item_text, encoding="us-ascii", method="xml"))

            # Decode list of bytes to a single string
            body_text = ' '.join([para.decode('utf-8') for para in paragraphs])

            # Body text still contains xml formatting, so remove all characters ]
            #   between brackets and double+ spaces.
            self.text = re.sub('<[^<]+>', "", body_text)
            self.text = re.sub(' +', ' ', self.text)
        except:
            # If XML could not be parsed, set text class variable to None to skip this
            #   paper.
            self.text = None
            error_msg = "Was not able to parse paper XML for paper " + paper["Title"]
            print(colored(error_msg, 'red'))

    def analyzeText(self):
        '''
        Finds name of dataset if included.
            - Scan for word dataset
            - Take preceeding words as long as there is no period and all
                previous words with capitalization are taken.
        Extracts any links in the body paragraphs.

        :params  N/A
        :return  N/A
        '''
        # Check if xml parser had access to paper
        if self.text is not None:
            sentences = self.text.replace('\n', ' ').split('.')

            #Find dataset names
            dataset_names = self.findDatasetName(sentences)

            # Find all links in text
            urls = self.findUrls()

            self.dataset_names = dataset_names
            self.urls = urls

        else:
            error = "The paper {} could not be opened as it is not open access".format(self.paper['Title'])
            print(colored(error, 'red'))
            self.dataset_names = []
            self.urls = []

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
        self.paper.update({'Dataset Name': self.dataset_names})
        self.paper.update({'Possible Dataset Links': self.urls})

    def APIRequest(self):
        '''
        Elsevier API handler for ScienceDirect. Supports both ScienceDirect Search V2 and
        Abstract Retrieval APIs (based on request_type param).

        Search V2: Uses supported PUT method with date, issue, and start_idx params.
        Abstract Retrieval: Uses GET method with DOI param.

        Headers need to include active API Key provided by the Elsevier website.
        https://dev.elsevier.com/apikey/manage

        :params   date: year for search (int)
                  issue: issue number for search (int)
                  start_idx: skip to this result (int)
        :return   (dict) contains search results for given page
        '''

        # SCIENCEDIRECT ABSTRACT RETRIEVAL
        while True:
            # Will try to access a website 5 times before exiting the function. Sometimes
            #  a bad response occurs so we test multiple times to ensure something failed.
            counter = 0
            try:
                key = self.apikey

                url = 'https://api.elsevier.com/content/article/doi/'

                url = url + self.doi

                headers = {"Accept": "text/xml",
                           "X-ELS-APIKey": key,
                           "content-type": "text/xml"}

                r = requests.get(url, headers=headers)
                time.sleep(.1)

                # return json.loads(str(r.text))
                return ElementTree.fromstring(r.content)

            # except json.JSONDecodeError:
            except Exception:
                # If query failed, wait for 1 second and try again
                counter += 1
                time.sleep(1)
                input('OOPS')

                # If enough failed attempts occur, exit function
                if counter > 5:
                    raise Exception("ScienceDirect has stopped responding.")


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(current_dir, '../../data/results_sciencedirect.json')
    with open(datapath, 'r') as f:
        contents = f.read()
        papers = json.loads(contents)

    # unique = { each['Title'] : each for each in papers }.values()
    # with open(datapath, 'w') as f:
    #     json.dump(list(unique), f, indent=4)

    input('Done')
    newpapers = []
    for paper in papers:
        sciencedirect = {'blah': 'blah', 'apikey': '8195584e9a1784037041888ba25292ee'}
        rand_list = ['blah', sciencedirect]
        config = [paper, rand_list]
        scraper = ScienceDirectScraper(config)
        newpaper = scraper.extract()

        newpapers.append(newpaper)

    with open('test_file.json', 'w') as f:
        json.dump(newpapers, f, indent=4)

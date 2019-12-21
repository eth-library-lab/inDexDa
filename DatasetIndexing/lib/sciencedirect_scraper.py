import re
import os
import time
import requests
from xml.dom import minidom
from xml.etree import ElementTree


class ScienceDirectScraper():
    def __init__(self, config):
        '''

        :param config: namedtuple containing information about which
                        topic papers will be found in
        '''
        self.paper = config[0]
        self.apikey = config
        input(config[1][1].apikey)
        self.doi = paper['DOI']

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

        paragraphs = []
        body_paragraphs = []

        with open('test.xml', 'wb') as f:
            mydata = ElementTree.tostring(article)
            f.write(mydata)

        # XML will only contain body text if it is deemed open-access. If it is
        #   not, return None.
        coredata = article.find('{http://www.elsevier.com/xml/svapi/article/dtd}coredata')
        openaccess = coredata.find('{http://www.elsevier.com/xml/svapi/article/dtd}openaccess')

        if not openaccess.text:
            print('Not open access')
            self.text = None
            exit()

        # If paper is open access, get body text by parsing xml.
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
        if self.text is None:
            exit()

        sentences = self.text.replace('\n', ' ').split('.')

        # Find all instances of 'Dataset' within the paper and compile the
        #   capitalized words in front of it to form a name
        dataset_name = []
        for sentence in sentences:
            words = sentence.split()
            if 'Dataset' in words:
                idx = words.index('Dataset')
                dataset_name.append(self.isPreviousWordCapitalized(words, idx))

        if dataset_name != []:
            # Remove duplicates
            dataset_name = list(set(dataset_name))
            # Add 'Dataset' to the name
            dataset_name = [name + ' Dataset' for name in dataset_name]
        else:
            dataset_name = []

        # Find all links in text
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', self.text)
        urls = self.checkUrls(urls)

        self.dataset_name = dataset_name
        self.urls = urls

    def isPreviousWordCapitalized(self, sentence, word_idx):
        '''
        Recursively checks if the previous word is capitalized and if so inserts
        it into the name list as the first entry.

        :params  sentence: list of words
                 word_idx: index of a word
        :return  name: string of dataset name
        '''
        name = []
        if sentence[word_idx - 1].istitle():
            name.append(sentence[word_idx - 1])
            name += self.isPreviousWordCapitalized(sentence, word_idx - 1)

        return ' '.join(name)

    def checkUrls(self, urls):
        '''
        Sometimes a new paragraph or space will cause regex to not identify the full
        url, so we check if the next character is a space or newline. Additionally
        the url may have extended past a bracket, so we trim it if we find one

        :params  url: string
        :return  extended_urls: string
        '''
        new_urls = []
        seps = ['(', ')', '{', '}', '<', '>', ' ']
        default_sep = seps[0]

        for url in urls:
            for sep in seps:
                url = url.replace(sep, default_sep)

            new_urls.append(url.split(default_sep)[0])

        return new_urls

    def updatePaper(self):
        '''
        Updates paper dict with dataset name and possible links to dataset if they were
        found in the paper pdf.

        :params  N/A
        :return  N/A
        '''
        self.paper.update({'Dataset Name': self.dataset_name})
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
    paper = {
        "Title": "Dataset for Evaluating the Accessibility of the Websites of Selected Latin American Universities",
        "DOI": '10.1016/j.dib.2019.105013',
        "Archive": "sciencedirect",
        "Prediction": "Dataset Detected"
        }

    apikey = '8195584e9a1784037041888ba25292ee'
    apiSciencedirect = ScienceDirectScraper(paper, apikey)
    paper = apiSciencedirect.extract()

    input(paper)

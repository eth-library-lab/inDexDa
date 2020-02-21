import os
import json
from rake_nltk import Rake


class generateKeywords():
    def __init__(self, filename):
        self.filename = filename

    def keywords(self):
        with open(self.filename, 'r') as f:
            contents = f.read()
            papers = json.loads(contents)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        arxiv_cat_dir = os.path.join(current_dir, '../data', 'arxiv_categories.json')
        with open(arxiv_cat_dir, 'r') as f:
            contents = f.read()
            arxiv_cat = json.loads(contents)

        self.new_papers = []
        for paper in papers:
            r = Rake(min_length=1, max_length=2)
            r.extract_keywords_from_text(paper['Abstract'])
            keywords = r.get_ranked_phrases()[0:5]

            if paper['Archive'] == 'arXiv':
                categories = paper['Category']
                for category in categories:
                    try:
                        keywords.append(arxiv_cat[category])
                    except:
                        print('OOPS')

            paper['Keywords'] = keywords
            self.new_papers.append(paper)

    def saveResults(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        outputdir = os.path.join(current_dir, 'test_results.json')
        with open(outputdir, 'w') as f:
            json.dump(self.new_papers, f, indent=4)



class indexDatasets():
    def __init__(self, filename):
        self.filename = filename

    def gatherDatasets(self):
        with open(self.filename, 'r') as f:
            contents = f.read()
            papers = json.loads(contents)

        datasets = {}
        for paper in papers:
            paper_datasets = paper['Dataset_Names']
            for paper_dataset in paper_datasets:
                # Check if dataset already in dict
                if paper_dataset in datasets:
                    # If yes, add keywords to the dict value
                    list1 = datasets[paper_dataset]
                    list2 = paper['Keywords']
                    fulllist = list1 + list2
                    datasets[paper_dataset] = fulllist
                else:
                    # If no, make new dict entry for dataset
                    datasets[paper_dataset] = paper['Keywords']

        outputdir = os.path.join(current_dir, 'dataset_names.json')
        with open(outputdir, 'w') as f:
            json.dump(datasets, f, indent=4)


class visualizeDatasets():
    def __init__(self, filename):
        self.filename = filename




if __name__ == '__main__':
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # datadir = os.path.join(current_dir, 'lib', 'test_file.json')
    # tester = generateKeywords(datadir)
    # tester.keywords()
    # tester.saveResults()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, 'test_results.json')
    tester = indexDatasets(datadir)
    tester.gatherDatasets()


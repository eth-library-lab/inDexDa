import json
from collections import namedtuple


Archive = namedtuple("Archive", "name query apikey start_year end_year")


def getInfoAboutArchivesToScrape():
    with open("args.json", 'r') as f:
        contents = f.read()
        try:
            input_args = json.loads(contents)

            archives = [item["archive"] for item in input_args["archive_to_scrape"]["archives"]]

            archive_info = []
            for item in input_args["archive_info"]:
                archive_info.append(Archive(name=item,
                                            query=input_args["archive_info"][item]["query"],
                                            apikey=input_args["archive_info"][item]["apikey"],
                                            start_year=input_args["archive_info"][item]["start_year"],
                                            end_year=input_args["archive_info"][item]["end_year"]))

        except ValueError:
            print('Not able to parse json file to dictionary.\n')

        return archives, archive_info
class ClassName(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg

def getInfoAboutNetworkParams():
    with open("args.json", 'r') as f:
        contents = f.read()
        try:
            input_args = json.loads(contents)
            epochs = input_args['epochs']
            batchSize = input_args['batchSize']

            params = [epochs, batchSize]

        except ValueError:
            print('Not able to parse json file to dictionary.\n')

        return params

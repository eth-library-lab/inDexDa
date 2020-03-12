import json
from tqdm import tqdm
from collections import namedtuple


Archive = namedtuple("Archive", "name query apikey start_year end_year")


def getInfoAboutArchivesToScrape():
    '''
    Gets info about which archives to scrape and info about these archives from
      args.json.
    '''
    with open("args.json", 'r') as f:
        contents = f.read()
        try:
            input_args = json.loads(contents)

            archives = [item for item in input_args["archive_to_scrape"]]

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


def getInfoAboutNetworkParams():
    '''
    Gets info about network parameters to use during training from args.json.
    '''
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


def removeDuplicates(list_of_dicts):
    '''
    Removes duplicates in a list of long dictionaries
    '''
    new_list = []
    for entry in tqdm(list_of_dicts):
        if entry not in new_list:
            new_list.append(entry)

    return new_list

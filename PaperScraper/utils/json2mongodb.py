import json
import re
from pymongo import MongoClient


index = "eth"
collection = "library_lab"
db = MongoClient('localhost', 27017)


def replace_values(dic_obj, q, replace_value):
    '''
    This function iterates recursive over a dictionary's values
    to replace them.
    '''
    for k, v in dic_obj.items():
        if isinstance(v, dict):
            dic_obj[k] = replace_values(dict(v), q, replace_value)

        if isinstance(v, str):
            dic_obj[k] = re.sub(q, replace_value, v)

    return dict(dic_obj)


def json2mongodb(dictionary):
    json_dictionary = dictionary

    # Some databases can't deal with NULL values:
    rec = replace_values(json_dictionary, ": null", ': ""')
    mongo_rec_id = db[index][collection].insert_one(rec).inserted_id
    '''
    From PyMongo documentation:
    'When a document is inserted a special key, "_id", is automatically
    added if the document doesn’t already contain an "_id" key.
    The value of "_id" must be unique across the collection.'

    I like to keep the original ID of the record, so I do something like:
    rec['_id'] = "liblab_" + rec['where_ever']['the']['original_ID']['is']
    If the record already exists, mongodb throws an exeption.
    You can avoid this by using "replace_one" instead of "insert_one".
    By using "replace_one" mongodb will insert the record if it's new
    or replace an existing one respectively.

    mongo_rec_id = db[index][collection].replace_one({'_id': rec['_id']}, rec, True)
    '''
    # print(mongo_rec_id)

if __name__ == '__main__':
    json2mongodb('/home/parker/code/datadex/PaperScraper/data/result.json')

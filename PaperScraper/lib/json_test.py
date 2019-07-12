from bs4 import BeautifulSoup
import json
import requests
import urllib.request


url = "https://arxiv.org/abs/1602.00020"
get_url = requests.get(url)
get_text = get_url.text
html = BeautifulSoup(urllib.request.urlopen(url),
                                     'html.parser')

content = html.findAll("div", {"id": "content"})
content = content[1]

title = content.find("h1", {"class": "title mathjax"}).text
authors = content.find("div", {"class": "authors"}).text
abstract = content.find("blockquote", {"class": "abstract mathjax"}).text
subject = content.find("span", {"class": "primary-subject"}).text
try:
    DOI = content.find("a", {"class": "link-https link-external"}).text
except:
    print("No DOI")

dict_title = {title.split(':')[0]: title.split(':')[1]}
dict_authors = {authors.split(':')[0]: authors.split(':')[1]}
dict_abstract = {abstract.split(':')[0]: abstract.split(':')[1].replace('\n','')}

# print(dict_title)
# print(dict_authors['Authors'])
# print(dict_abstract)
print(subject)
print(DOI)

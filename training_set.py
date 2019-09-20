import json

titles = []
abstracts = []
with open('data/true_positives.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if((i + 3) % 3 == 0):
            titles.append(line)
        if((i + 2) % 3 == 0):
            abstracts.append(line)

papers = []
for idx in range(len(titles)):
    papers.append({'Title': titles[idx], 'Abstract': abstracts[idx]})


with open('data/dataset.json', 'w') as f:
    json.dump(papers, f, indent=4)

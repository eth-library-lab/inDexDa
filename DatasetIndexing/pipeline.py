import os
import json


### STEPS
# 1) Get full papers for all papers BERT predicts point towards a new dataset
#       a) Make a generalized method which first searches ArXiv, then ScienceDirect
#       b) Grabs first xml formatted full paper it can
#       c) Parse xml to get the main body texts
# 2) Search through parsed xml to get link
#       a) Search abstract first
#       b) Search each subsequent body paragraph next
#       c) Store all links it finds, except obvious xml formatting links and emails

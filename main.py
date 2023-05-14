## DEPENDENCIES:

import pandas as pd ## Data Storage, Manipulationm
import numpy as np ## Math, np arrays, vectorization

## AutoTokenizer: Converts English strings into numeric codes, then usable with BERT NLP model (search word tokenizer, character tokenizer, subword tokenizer stuff)
## AutoModelForSequenceClassification: Architechure used to load in NLP model from HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification

## PyTorch: Deep Learning, working with NLP models, models written in PyTorch, specifically using ArgMax to extract highest sequence result (Sentiment Analysis)
import torch

## BeautifulSoup4: Allows to process HTML of webpage, scrape info
## Requests: Allows to fetch HTML pages of links
from bs4 import BeautifulSoup
import requests

## Create regex function, extract specific info from processed HTML page
import re

## Model Instance

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') ## Converting strings to tokens according to BERT Model specifications
model=AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment') ## Converting tokens into predicted outcome using BERT Model

#### MORE EXPLANATION ON THIS LATER:


## Building review scraping - making requests to Yelp! site
req = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')  ## Aquiring HTML
soup = BeautifulSoup(req.text, 'html.parser')  ## Object representing HTML page of link
regex = re.compile('.*comment.*')  ## Using regex expression to find all classes that are comments on restaurant (to use for customer sentiment analysis)
results = soup.find_all('p', {'class':regex})  ## Isolating p tags under all possible tags that consitute as comments on Yelp! page
reviews = [result.text for result in results]  ## Iterating through p tags, extracting english text, saving in list, extraction done


## Load collected reviews into df, score
df = pd.DataFrame(np.array(reviews), columns=['reviews'])

def sentiScore(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

## Vectorizing
df['senti'] = df['reviews'].apply(lambda x: sentiScore(x[:512])) ## Apply is a mapping method, using lambda to pass in. NLP pipeline is limited to # of tokens acceptable at once
                                                                ## We will only analyze first 511 words from the review
print(df)
print(df['senti'].mean())

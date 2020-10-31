import nltk
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re, string
nltk.download('stopwords')
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import random

def remove_noise(tweet_token, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_token):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "",
                       token)  # link
        token = re.sub("(?:@[A-Za-z0-9_]+)", "", token)  # tag

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_token_list):
    for tokens in cleaned_token_list:
        for token in tokens:
            yield token

def get_tweet_for_model(cleaned_token_list):
    for tweet_tokens in cleaned_token_list:
        yield dict([token, True] for token in tweet_tokens)


def tweets_dic(collections_list):
    tweets_dictionary = {}
    for a_collection in collections_list:
        result = a_collection.aggregate([
            {
                '$match': {
                    'full_text': {
                        '$in': [
                            re.compile(r"God"), re.compile(r"god"), re.compile(r"jesus"), re.compile(r"Bible"),
                            re.compile(r"bible"), re.compile(r"Church"), re.compile(r"church"), re.compile(r"Dio"),
                            re.compile(r"Dios"), re.compile(r"Jesus"), re.compile(r"Gesu"), re.compile(r"Gesù"),
                            re.compile(r"gesu"), re.compile(r"gesù"), re.compile(r"Christian"),
                            re.compile(r"Christianity"), re.compile(r"christian"), re.compile(r"christianity"),
                            re.compile(r"Holy Spirit"), re.compile(r"faith"), re.compile(r"Faith"), re.compile(r"amen"),
                            re.compile(r"Amen"), re.compile(r"Christ"), re.compile(r"christ"), re.compile(r"gospel"),
                            re.compile(r"Gospel"), re.compile(r"Good News"), re.compile(r"grace"), re.compile(r"Grace"),
                            re.compile(r"Worship"), re.compile(r"worship"), re.compile(r"catholic"),
                            re.compile(r"Catholic"), re.compile(r"Pentecostal"), re.compile(r"pentecostal"),
                            re.compile(r"pray"), re.compile(r"Pray"), re.compile(r"prega"), re.compile(r"Prega"),
                            re.compile(r"pregh"), re.compile(r"Pregh"), re.compile(r"prego"), re.compile(r"Prego")
                        ]
                    }
                }
            }, {
                '$project': {
                    'id': 1,
                    'user.id': 1,
                    'full_text': 1,
                    'entities.hashtags': 1,
                    'lang': 1
                }
            }, {
                '$addFields': {
                    'user_id': '$user.id',
                    'hashtags': '$entities.hashtags.text'
                }
            }, {
                '$project': {
                    'user': 0,
                    'entities': 0
                }
            }, {
                '$sort': {
                    'lang': -1
                }
            }
        ])

        tweets_dictionary[str(a_collection)[-14:-2]] = list(result)

    return tweets_dictionary

def get_tweet_text(a_dictionary):
    tweets_dict_full_text = {}
    for a_key in a_dictionary:
        tweets_dict_full_text[a_key] = []
        for i in range(len(a_dictionary[a_key])):
            tweets_dict_full_text[a_key].append(a_dictionary[a_key][i]['full_text'])
    return tweets_dict_full_text

def get_christian_tweets(full_text_dictionary, christian_search):
    christian_tweets_dictionary = {}
    for a_key in full_text_dictionary:
        christian_tweets_dictionary[a_key] = []
        for a_tweet in full_text_dictionary[a_key]:
            for a_word in christian_search:
                if a_word in a_tweet.lower().split():
                    christian_tweets_dictionary[a_key].append(a_tweet)
                    break
                else:
                    continue
        christian_tweets_dictionary[a_key] = set(christian_tweets_dictionary[a_key])
    return christian_tweets_dictionary

def count_sentiment(sentiment_list):
    num_pos = 0
    num_neg = 0
    for a_sentiment in sentiment_list:
        if a_sentiment == 'Positive':
            num_pos += 1
        else:
            num_neg += 1
    fixing_factor = (37/158 + 46/198 + 84/176)/3
    new_positive = int(num_pos + num_neg * fixing_factor)
    new_negative = int((1 - fixing_factor) * num_neg)
    sentiment_count_dict = {'pos' : new_positive, 'neg' : new_negative}
    return sentiment_count_dict

def get_sentiment_dict(christian_tweets_dict, classifier):
    sentiment_dictionary = {}
    for a_key in christian_tweets_dict:
        sentiment_list = []
        for tweet in christian_tweets_dict[a_key]:
            Tweet_cleaned = remove_noise(word_tokenize(tweet))
            sentiment_list.append(classifier.classify(dict([token, True] for token in Tweet_cleaned)))
        sentiment_dictionary[a_key] = count_sentiment(sentiment_list)
    return sentiment_dictionary




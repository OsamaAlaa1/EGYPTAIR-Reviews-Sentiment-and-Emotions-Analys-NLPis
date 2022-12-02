# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:21:43 2022

@author: Osama Ala'a
"""
#_______________________ 1. Extract data Using BeautifulSoup _______________________
import requests 
from bs4 import BeautifulSoup
import pandas as pd

#---------------------- skytrax section --------------------------------

base_url = "https://www.airlinequality.com/airline-reviews/egyptair"
pages = 4            # number of pages 
page_size = 100      

reviews = []        # 100*4 = 400 review  

# for i in range(1, pages + 1) to iterate over the pages
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")


#-------------------------- trustpilot section ----------------------------------

pages = 12  # number of pages 
pilot_reviews = []

# for i in range(1, pages + 1) to iterate over the pages
for i in range(1, pages + 1):
    
    # Create URL to collect links from paginated data
    url = f"https://www.trustpilot.com/review/egyptair.com?page={i}&sort=recency"

    print(f"Scraping page {i}")

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("p", {"class": "typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn"}):
        pilot_reviews.append(para.get_text())
    

    print(f"   ---> {len(pilot_reviews)} total reviews")

          
#_______________________ 2. turn extracted data into csv _______________________

df_st = pd.DataFrame()
df_st["reviews"] = reviews
df_st.to_csv("EGYPTAIR_reviews_skytrax.csv")

df_tp = pd.DataFrame()
df_tp["reviews"] = pilot_reviews
df_tp.to_csv("EGYPTAIR_reviews_trustpilot.csv")


#_______________________ 3. read and load csv file _______________________

#read the csv file for EGYPTAIR reviews from skyrax and take a look 
df_st = pd.read_csv("EGYPTAIR_reviews_skytrax.csv", index_col= 0)
df_st.head()

#read the csv file for EGYPTAIR reviews from trustpilot and take a look 
df_tp = pd.read_csv("EGYPTAIR_reviews_trustpilot.csv", index_col= 0)
df_tp.head()

#_______________________ 4. text Cleaning function _______________________

import string   # we need it for Punctuations removal
from stop_words import get_stop_words   # or we can use from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize # it turn text to list but more faster 

def sentiment_clean_text(text):
    '''
    this function take text and clean it  

    Parameters
    ----------
    text : string 

    Returns
    -------
    text_list : string after cleaning.

    '''


    # A. first step remove useless text data (if there are any) note: i don't need data before | so I will remove it 
    if '|' in text:
        text =  text.split('|')[1]   
       
    # B. second turn letters into lowercase 
    text = text.lower()
        
    # C. third remove all Punctuations.
    text = text.translate(str.maketrans('','',string.punctuation))
    
    return text

def emotion_clean_text(text):
    '''
    this function take text and clean it then turn it to list of words 

    Parameters
    ----------
    text : string 

    Returns
    -------
    text_list : list of text words after cleaning.

    '''
        
    # D. forth step Tokenization and stop words 
        
    # Tokenizaiton: turning string into list of words.
    # Stop words: words without meaning for sentiment analysis.

        
    # Tokenizaiton
    text = word_tokenize(text,"english")
    
    # handeling the stop words but what are the stop words 
    stop_words = get_stop_words('english') #or we can use stop_words = stopwords.words('english')

    # Removing stop words from the tokenized words list
    text_list = []
    
    for word in text:
        if word not in stop_words:
            text_list.append(word)

       
    # return the list of words
    return text_list

#___________________________ 5. Emotions detection ____________________________

def emotion_maping (file,di): 
    
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        di[word] = emotion
            
    return di


#_____________________________ 6. Sentiment Analysis Vader Model (doesn't account for words relationships - use Roberta Analysis instead) ________________________

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analyze(text):
    
    scores = SentimentIntensityAnalyzer().polarity_scores(text) # return dictionary of scores
    
    if (scores['neg'] > scores['pos']):
        return 0
    
    else:
        return 1



#___________________________ 7. Main _______________________________________

from collections import Counter

# define needed data structures
cleaned_text = ""
temp_emotion_list = []
score = 0
emotion_dict = {}
words_score_dict = {}
moods_list_st = []
moods_list_tp = []

# get the emotion dictionary ready
emotion_file = open('emotions.txt','r',encoding='utf-8') 
emotion_dict = emotion_maping(emotion_file,emotion_dict)
emotion_file.close()

# ------------------------ loop for the skytrx Dataframe ------------------------------

# loop for all reviews in Skytrax dataFrame  
for i in range(len(df_st)):
    
    # get the review of index i
    text = str(df_st['reviews'][i])
    
    # step 1: let's clean the text and assign cleaned list to dataFrame 
    # simple clean 
    cleaned_text= sentiment_clean_text(text)
    
    #Step 2: sentiment Analysis
    score = sentiment_analyze(cleaned_text)
    moods_list_st.append(score)

    
    # Step 3: advanced clean for emotions
    cleaned_text_list = emotion_clean_text(cleaned_text)
    df_st['reviews'][i] = cleaned_text_list
    

    # Step 4: emotion list builder
    for word in emotion_dict.keys():
        if word in cleaned_text_list:
            temp_emotion_list.append(emotion_dict[word])    

# now let's create new column for moods for skytrax
df_st['mood'] = moods_list_st


# ------------------------ loop for the trustpilot Dataframe ------------------------------

# loop for all reviews in trustpilot dataFrame  
for i in range(len(df_tp)):
    
    # get the review of index i
    text = str(df_tp['reviews'][i])
    
    # step 1: let's clean the text and assign cleaned list to dataFrame 
    # simple clean 
    cleaned_text= sentiment_clean_text(text)
    
    #Step 2: sentiment Analysis
    score = sentiment_analyze(cleaned_text)
    moods_list_tp.append(score)

    
    # Step 3: advanced clean for emotions
    cleaned_text_list = emotion_clean_text(cleaned_text)
    df_tp['reviews'][i] = cleaned_text_list
    

    # Step 4: emotion list builder
    for word in emotion_dict.keys():
        if word in cleaned_text_list:
            temp_emotion_list.append(emotion_dict[word])  


# now let's create new column for moods trust pilot dataframe
df_tp['mood'] = moods_list_tp


#____________________________ Plot for sentiment Vader scores ____________________________________


# Step 5: plot sentiment and emotions 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# get summary dictionary for sentiment 
moods_list_all = moods_list_st + moods_list_tp
mmod_summary = dict(Counter(moods_list_all))

# rename dictionary keys before ploting 
mmod_summary['positive'] = mmod_summary.pop(1)
mmod_summary['negative'] = mmod_summary.pop(0)

fig, ax = plt.subplots()
ax.bar(mmod_summary.keys(), mmod_summary.values())

# label the plot 
plt.xlabel('Sentiments')
plt.ylabel('Frequency')
plt.savefig('Sentiments.png')
plt.show()

#____________________________ Plot for emotions ____________________________________
# turn the emotion list into dictionary 
words_score_dict = dict(Counter(temp_emotion_list))

# now let's go more Specifically and just choose emotions with freq more than 10
more_than_15_emotions = dict( (key, value) for (key, value) in words_score_dict.items() if value > 15 )


fig, ax = plt.subplots()
#ax.bar(words_score_dict.keys(), words_score_dict.values())
ax.bar(more_than_15_emotions.keys(), more_than_15_emotions.values())

# the fit the text undre graph 
fig.autofmt_xdate()

# change the width and length of plot 
fig.set_figwidth(20)
fig.set_figheight(10)

# label the plot 
plt.xlabel('Emotions')
plt.ylabel('Frequency')

#save the graph and show it 
plt.savefig('More_than_15_emotions.png')
plt.show()



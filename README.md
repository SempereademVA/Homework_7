

```python
%matplotlib inline
# Dependencies
import tweepy
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from citipy import citipy
import seaborn as sns
from config import consumer_key, consumer_secret, access_token, access_token_secret


# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
#number_of_pages = 5
#Target Users
user_lst =('BBCNews', 'CBSNews','cnn','FoxNews', 'nytimes')
number_of_pages = len(user_lst)
```


```python
now = time.strftime("%c")
the_date = time.strftime("%x")
```


```python
#Initialize lists for the dictionary
source_list = []
tweet_texts = []            # Tweet Texts
tweet_times = []            # List to hold tweet timestamps
tweets_ago_list = []        # List for the data dictionary and dataframe

                            # Variables for holding sentiments from VADER
compound_list = []
positive_list = []
negative_list = []
neutral_list = []



for i in range(len(user_lst)):  # Loop through all the news sites

    target_user = user_lst[i]
    counter = 1                 # Counter to keep track of tweets ago
    oldest_tweet = None         # Variable for max_id



    for x in range(number_of_pages):       # Loop through tweet pages
        public_tweets = api.user_timeline(target_user, page=x)  # Get all tweets from home feed (for each page specified)
        
        for tweet in public_tweets:                             # Loop through all tweets

            print(tweet["text"])
        
            tweet_texts.append(tweet["text"])                   # Store Tweet in Array
        
            raw_time = tweet["created_at"]                      # Store Time in Array 
            print(raw_time)
            tweet_times.append(raw_time)                        # Append to time array for dictionary
        
                                                                # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])   # Use VADER
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
        
                                      
            oldest_tweet = tweet['id'] - 1                      # Get Tweet ID, subtract 1, and assign to oldest_tweet

                                                            # Add each value to the appropriate list
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            tweets_ago_list.append(counter)
            source_list.append(target_user)
        
            counter += 1                                        #Increment counter for last tweet
        
```


```python
news_dic={'Source': source_list, 'Tweet':tweet_texts, 'Time': tweet_times,
          'Compound': compound_list, 'Positive':positive_list, 'Neutral':neutral_list ,'Negative': negative_list,
          'Tweets Ago': tweets_ago_list  }
```


```python
news_df = pd.DataFrame.from_dict(news_dic)
```


```python
news_df.head()
```


```python
news_df.to_csv('example.csv')
```


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# plot
sns.set_style("whitegrid", {'axes.grid' : True})


#sns.regplot(x="Tweets Ago", y="Compound", data=df, ax=ax, fit_reg=False)
g=sns.lmplot( x="Tweets Ago", y="Compound", data=news_df, fit_reg=False, hue='Source', 
             legend=True, size=12, aspect=1.5,scatter_kws={"s": 500})
sns.despine()
g.set(ylim=(-1, None), xlim=(0,None))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Sentiment Analysis of Tweets ' + the_date, fontsize=34)
plt.savefig('scatter_sentiment.png')
```


```python
median_array = [('Aggregate', ['Median'])]

for i in range(len(user_lst)):
    median_df=news_df[news_df['Source'] == user_lst[i]]
    x_median = np.mean(median_df['Compound'])
    x_median = round(x_median,4)
    temp = [x_median]
    temp_list = (user_lst[i],temp)
    median_array.append(temp_list)
    temp_list = ()
```


```python
median_df = pd.DataFrame.from_items(median_array)
```


```python
median_df
```


```python
#sns.set_style("whitegrid")
sns.set_style("whitegrid", {'axes.grid' : True})
sns.set(font_scale=2.5)

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)

ax = sns.barplot(data=median_df)
ax.set(ylim=(-0.4, 0.4), xlim=(0,None))
ax.set_title("Overall Media Sentiment Based on Twitter " + the_date,fontsize=30)
ax.set_xlabel("Media Source", size = 35)
ax.set_ylabel("Tweet Polarity", size = 35)
plt.savefig('polarity_plot.png')
```

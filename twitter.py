# -*- coding: utf-8-*-

import tweepy
import csv
import pandas as pd

consumerKey='uRDuync3BziwQnor1MZFBKp0x'
consumerSecret='t8QPLr7RKpAg4qa7vth1SBsDvoPKawwwdEhNRjdpY0mfMMdRnV'
AccessToken='14366551-Fga25zWM1YefkTb2TZYxsrx2LVVSsK0uSpF08sugW'
AccessTokenSecret='3ap8BZNVoBhE2GaMGLfuvuPF2OrHzM3MhGuPm96p3k6Cz'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(AccessToken, AccessTokenSecret)

api = tweepy.API(auth, wait_on_rate_limit=True)

api.update_status("Using Python for downloading Tweets on 06 Jul")


csvFile = open('yash.csv','w')

csvWriter = csv.writer(csvFile)
for tweet in tweepy.Cursor(api.search, q="@yash_j1301", count=100, lang="en", since = "2019-01-01").items():
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    
csvFile = open('ua.csv','a')
csvWriter = csv.writer(csvFile)
for tweet in tweepy.Cursor(api.search, q="#unitedAirlines", count=100, lang="en", since = "2019-01-01").items():
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
                 
csvFile = open('icc.csv','a')
csvWriter = csv.writer(csvFile)
for tweet in tweepy.Cursor(api.search, q="#ICC", count=100, lang="en", since = "2019-01-01").items():
    print(tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


search_words = "icc"
date_since = "2019-01-01"

tweets = tweepy.Cursor(api.search, q=search_words, lang="en", since=date_since).items(5)

tweets


for tweet in tweets:
    print(tweet.text)

tweets = tweepy.Cursor(api.search, q=search_words, lang="en", since=date_since).items(5)
[tweet.text for tweet in tweets]


new_search = search_words + "-filter:retweets"
new_search

tweets = tweepy.Cursor(api.search, q=new_search, lang="en", since=date_since).items(5)
[tweet.text for tweet in tweets]


tweets = tweepy.Cursor(api.search, q=search_words, lang="en", since=date_since).items(10)  

user_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
user_locs


tweet.text = pd.DataFrame(data=user_locs, columns=['twitter name', 'location'])
tweet.text

tweets = tweepy.Cursor(api.search, q="Analytics", lang="en", since="2019-01-01").items(10)
tweets
for tweet in tweets:
    print(tweet.text)

user_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets]
user_locs 

new_search = "data + analytics -filter:retweets"
date_since = "2019-01-30"
tweets = tweepy.Cursor(api.search, q=new_search, lang="en", since=date_since).items(10)  
all_tweets = [[tweet.text for tweet in tweets]]
all_tweets[:5]
len(all_tweets) 

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
    
for tweet in public_tweets:
    print(tweet.created_at)
    print(tweet.user.screen_name)
    print(tweet.user.location)
    

user_tweets = api.user_timeline()

for tweet in user_tweets:
    print(tweet.text)
    
for tweet in user_tweets:
    print(tweet.created_at)
    print(tweet.user.screen_name)
    print(tweet.user.location)

user_tweets = api.user_timeline(count=10)

for tweet in user_tweets:
    print(tweet.text, "\t", tweet.created_at, "\t" ,tweet.user.screen_name, "\t", tweet.user.location)


name = "india"
tweetCount=20
results = api.user_timeline(id=name, count=tweetCount)
for tweet in results:
    print(tweet.text, "\n")


query ="Cricket"
language = "en"
results = api.search(q=query, lang=language)
for tweet in results:
    print( tweet.user.screen_name, "\t", tweet.text ,"\n" )


#Assignment - spatial Graph on where ICC was mentioned in the world
#Assignment - Sentimental Analysis on tweet, find overall opinion on GST in india
#Assignment - Social graph on most popular users that tweet about ICC cricket


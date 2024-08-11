import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "coingecko app"
tweets = []
limit = 100

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    #print(vars(tweet))
    #break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.content])

print(tweets)
#tweet.date, tweet.user.username, 
#df = pd.DataFrame(tweets, columns = ['Date', 'User','Tweet'])
#print(df)

# ------



import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

#----THE SCRAPING TWEETS PART
query = "coingecko app"
tweets = []
limit = 100

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.content])

#print(tweets)
#tweets in the format of [['tweet'],['tweet']]

# this variable will stored all the processed tweets in a single-dimension array
tweet_proc = []

# preprocess tweet because the model only takes in @user as mention and http for links
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

for i in range(len(tweets)):
    tweet_proc.append(preprocess(tweets[i][0]))

#Display tweets
df1 = pd.DataFrame(tweet_proc, columns= ['Tweets'])
print(df1)

#----THE MACHINE LEARNING PART
# bring in the training model
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

#array to store scores_prob 
scores_prob = []

for i in range(len(tweet_proc)):
# tokenize input text, ie convert text to integers 
    encoded_tweet = tokenizer(tweet_proc[i], return_tensors = 'pt')
# another way of writing model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)
# obtain the scores of sentiment analysis computation
    scores = output[0][0].detach().numpy()
# convert the scores into probability
    scores_prob.append(softmax(scores))

print(scores_prob)

#putting in labeled columns and outputting the aggregated mean. DF2 will show the overall sentiment. 
df = pd.DataFrame(scores_prob, columns = ['Negative','Neutral', 'Positive'])
print(df)
df2 = df.mean(axis=0)
print(df2)
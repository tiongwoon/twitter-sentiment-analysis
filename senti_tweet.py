from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np

tweet = ""

# preprocess tweet because the model only takes in @user as mention and http for links
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

tweet_proc = preprocess(tweet)

# bring in the training model
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# labels for output 
labels = ['Negative', 'Neutral', 'Positive']

# tokenize input text, ie convert text to integers 
encoded_tweet = tokenizer(tweet_proc, return_tensors = 'pt')

# another way of writing model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

# obtain the scores of sentiment analysis computation
scores = output[0][0].detach().numpy()

#print (scores)

# convert the scores into probability
scores_prob = softmax(scores)
#print(scores_prob)

# output data in readable format 
for i in range(len(scores)):
    l = labels[i]
    s = scores_prob[i]
    print(l,s)
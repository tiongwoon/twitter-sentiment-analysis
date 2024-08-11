# Twitter sentiment analysis 

A simple script that takes in a keyword, scrapes twitter to get the 100 most recent tweets, and feeds them into a sentiment analysis model. It then outputs
- Aggregated mean and split into Negative, Neutral, Positive buckets
- Overall sentiment score

The model used is from https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

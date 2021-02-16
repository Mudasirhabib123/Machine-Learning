from nltk import pos_tag,word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd 


speach = 'Fahad love walking indoor and the friend of Mudasir'
print(pos_tag(word_tokenize(speach)))
tweets = [
            "I am eating a burrito for breakfast",
            "Political science is an amazing field",
            "San Francisco is an awesome city"
            ]

tagged_tweets = []
for tweet in tweets:
    tagged = pos_tag(word_tokenize(tweet))
    tagged_tweets.append(tw for tag,tw in tagged)

binarizer = MultiLabelBinarizer()
string = binarizer.fit_transform(tagged_tweets)

df = pd.DataFrame(string, columns = binarizer.classes_)
# print(binarizer.classes_)
# print(string)
print(df)
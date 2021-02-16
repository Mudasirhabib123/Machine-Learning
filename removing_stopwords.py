from nltk.corpus import stopwords

tokenized_words = [
                    'i',
                    'am',
                    'going',
                    'to',
                    'go',
                    'to',
                    'the',
                    'store',
                    'and',
                    'park'
                    ]


stopwords = stopwords.words('english')

string = [word for word in tokenized_words if word not in stopwords]
print(string)
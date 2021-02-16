from nltk.stem.porter import PorterStemmer

tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

porter = PorterStemmer()

stem_words = [porter.stem(w) for w in tokenized_words]

print(stem_words)
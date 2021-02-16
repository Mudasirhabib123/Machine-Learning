from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger,TrigramTagger

sentence = brown.tagged_sents(categories = 'news')

train = sentence[:4000]
test = sentence[4000:]

unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff= unigram)
trigram = TrigramTagger(train, backoff= bigram)

print(trigram.evaluate(test))
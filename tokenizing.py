from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import data

# data.path += ['/home/mudasir_habib/SKLearn/venv/ntlk_data']

string = "The science of today is the technology of tomorrow"
print(string)

string = word_tokenize(string)
print(string)

string = "The science of today is the technology of tomorrow. tommorow is today"
string = sent_tokenize(string)

print(string)
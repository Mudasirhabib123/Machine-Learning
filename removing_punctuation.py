import unicodedata
import sys

text_data = [
            'Hi!!!! I. Love. This. Song....',
            '10000% Agree!!!! #LoveIT',
            'Right?!?!'
            ]

punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

string = [string.translate(punctuation) for string in text_data]

print(string)
import nltk.corpus
from keyphrase.keyphrase import KeyPhrase

common_words = KeyPhrase( text = nltk.corpus.gutenberg.raw(fileids='austen-emma.txt')).tokens(500)

# Read a file, strip it from common words, break it in partitions of 1000 words long and score the individual words
kp = KeyPhrase(file="scripts/script.txt")\
            .exclude(common_words["token"].tolist())\
            .partition(1000)\
            .score()

# The KeyPhrase object creates a df that simplifies automatic analyses of text
# functions either modify this df (by adding or changing one of its columns)
# or query it like trigrams, bigrams and tokens do..
# print(kp.df)

# print the top 2 trigrams per partition
print(kp.trigrams(2))

# print the top 3 tokens per partition
print(kp.tokens(3))

# print the top 1 bigrams per partition
print(kp.bigrams(1))
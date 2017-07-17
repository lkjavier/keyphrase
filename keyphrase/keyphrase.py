import pandas as pd
import nltk
import string
import math

class KeyPhrase(object):
    def __init__(self, file=None, df=None, text=None):
        if file != None:
            text = open(file).read().lower()
        if text != None:
            self.df = pd.DataFrame([s.lower() for s in nltk.word_tokenize(text)])
            self.df.columns = ["token"]
            self.df["index"] = range(0, len(self.df))
            self.df["active"] = True
            self.df["partition"] = 0
            self.df["score"] = 0
            self.df["is_punctuation"] = [token in string.punctuation for token in self.df.token]

    def exclude(self, tokens):
        self.df["active"] = [(token not in tokens) for token in self.df["token"]]
        return self

    def partition(self, size=None):
        self.df["partition"] = [math.floor(x / size) for x in self.df.index]
        return self

    def score(self):
        token_df = self.df.groupby(["token", "partition"]).agg({
            'active': 'count'}) \
            .rename(columns={'active': 'score'}) \
            .reset_index()
        self.df = self.df.drop('score', 1)
        self.df = self.df.merge(token_df, on=["token", "partition"]).sort_values("index").reset_index(drop=True)
        return self

    def tokens(self, n=100):
        return self.df[(self.df.active == True) & (self.df.is_punctuation == False)] \
            .groupby(["token", "partition"]).agg({
            'score': 'sum',
            'active': 'count'}) \
            .rename(columns={'active': 'count'}) \
            .reset_index() \
            .sort_values(["partition", "count"], ascending=[True, False]) \
            .groupby("partition") \
            .head(n)

    def bigrams(self, n=5):
        self.df["bigrams"] = [self.__n_gram(index, 2) for index in self.df.index]
        self.df["bigrams_score"] = [self.__n_gram_score(index, 2) for index in self.df.index]
        return self.df[(self.df.active == True) & (self.df.is_punctuation == False)] \
            .groupby(["bigrams", "partition"]).agg({
            'bigrams_score': 'sum'}) \
            .rename(columns={'bigrams_score': 'score'}) \
            .reset_index() \
            .sort_values(["partition", "score"], ascending=[True, False]) \
            .groupby("partition") \
            .head(n)

    def trigrams(self, n=5):
        self.df["trigrams"] = [self.__n_gram(index, 3) for index in self.df.index]
        self.df["trigrams_score"] = [self.__n_gram_score(index, 3) for index in self.df.index]
        return self.df[(self.df.active == True) & (self.df.is_punctuation == False)] \
            .groupby(["trigrams", "partition"]).agg({
            'trigrams_score': 'sum'}) \
            .rename(columns={'trigrams_score': 'score'}) \
            .reset_index() \
            .sort_values(["partition", "score"], ascending=[True, False]) \
            .groupby("partition") \
            .head(n)

    def __n_gram(self, index, n):
        if sum(~self.df.loc[index:index + n - 1].is_punctuation) == n and self.df.loc[index + n - 1].active:
            return " ".join(self.df.loc[index:index + n - 1, "token"].tolist())

    def __n_gram_score(self, index, n):
        if sum(~self.df.loc[index:index + n - 1].is_punctuation) == n and self.df.loc[index + n - 1].active:
            return sum(self.df.loc[index:index + n - 1, "score"].tolist())
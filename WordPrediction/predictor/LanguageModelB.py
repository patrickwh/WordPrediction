
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
import re

class LanguageModelB:
    
    def __init__(self):
        self.train()
        
    def train(self):
        print('Training...')
        
        tri_fd = nltk.FreqDist()
        tri_cfd = nltk.ConditionalFreqDist()
        bi_fd = nltk.FreqDist()
        bi_cfd = nltk.ConditionalFreqDist()
        uni_fd = nltk.FreqDist()
        
        genres = ['adventure']

        for genre in genres:
            
            corpus = brown.tagged_words(categories = genre)
            size = int(len(corpus) * 0.90)
            corpus = corpus[:size]
            trigrams = nltk.trigrams(corpus)
            bigrams = nltk.bigrams(corpus)

            for ((word2, tag2), (word1, tag1), (word0, tag0)) in trigrams:
                tri_fd[word2, word1, word0] += 1
                tri_cfd[word2, word1][word0] += 1      

            for ((word1, tag1),(word0, tag0)) in bigrams:
                bi_fd[word1, word0] += 1
                bi_cfd[word1][word0] += 1    

            for ((word0, tag0)) in corpus:
                uni_fd[word0] += 1      
            
        # n-gram probability distributions
        self.tri_cpd = nltk.ConditionalProbDist(tri_cfd, nltk.LaplaceProbDist)
        self.tri_pd = nltk.SimpleGoodTuringProbDist(tri_fd)
        
        self.bi_cpd = nltk.ConditionalProbDist(bi_cfd, nltk.LaplaceProbDist)
        self.bi_pd = nltk.SimpleGoodTuringProbDist(bi_fd)
        
        self.uni_pd = nltk.SimpleGoodTuringProbDist(uni_fd)

        print('Done!')
        
    def simple_linear_interpolation(self, w2, w1):
        
        alpha = 0.6
        beta = 0.25
        gamma = 0.15

        best = 0
        word = 'default'
        
        for w0 in self.uni_pd.samples():
            try:
                tri = self.tri_cpd[w2, w1].prob(w0)
            except:
                tri = self.tri_pd.prob((w2, w1, w0))
            try:
                bi = self.bi_cpd[w1].prob(w0)
            except:
                bi = self.bi_pd.prob((w1, w0))
                
            uni = self.uni_pd.prob(w0)
            tmp = alpha * tri + beta * bi + gamma * uni
            if (tmp > best):
                word = w0
                best = tmp

        return word
    
    def get_score(self, w2, t2, w1, t1, w0, t0):
        
        alpha = 0.6
        beta = 0.25
        gamma = 0.15
        
        try:
            tri = self.tri_cpd[w2, w1].prob(w0)
        except:
            tri = self.tri_pd.prob((w2, w1, w0))
        try:
            bi = self.bi_cpd[w1].prob(w0)
        except:
            bi = self.bi_pd.prob((w1, w0))
                
        uni = self.uni_pd.prob(w0)
        tmp = alpha * tri + beta * bi + gamma * uni
        
        return tmp

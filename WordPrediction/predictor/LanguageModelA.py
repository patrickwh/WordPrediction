
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
import re

class LanguageModelA:
    
    def __init__(self):
        self.train()
        
    def train(self):
        print('Training model A...')

        tri_cfd = nltk.ConditionalFreqDist()
        bi_cfd = nltk.ConditionalFreqDist()
        uni_fd = nltk.FreqDist()
        
        wordtag_tri_cfd = nltk.ConditionalFreqDist()
        wordtag_bi_cfd = nltk.ConditionalFreqDist()
        wordtag_uni_fd= nltk.FreqDist()
        
        genres = ['news', 'religion', 'reviews', 'romance', 'science_fiction']
        
        for genre in genres:

            corpus = brown.tagged_words(categories = genre)
            size = int(len(corpus) * 0.90)
            corpus = corpus[:size]
            trigrams = nltk.trigrams(corpus)
            bigrams = nltk.bigrams(corpus)
            
            for ((word2, tag2), (word1, tag1), (word0, tag0)) in trigrams:  
                tri_cfd[word2, word1][word0] += 1      
                wordtag_tri_cfd[word2, tag2, word1, tag1][word0] += 1

            for ((word1, tag1),(word0, tag0)) in bigrams:
                bi_cfd[word1][word0] += 1    
                wordtag_bi_cfd[word1, tag1][word0] += 1

            for ((word0, tag0)) in corpus:
                uni_fd[word0] += 1      
                wordtag_uni_fd[word0, tag0] += 1

        # n-gram probability distributions
        self.tri_cpd = nltk.ConditionalProbDist(tri_cfd, nltk.ELEProbDist)
        self.bi_cpd = nltk.ConditionalProbDist(bi_cfd, nltk.ELEProbDist)
        self.uni_pd = nltk.ELEProbDist(uni_fd)
        
        # POS n-gram
        self.wordtag_uni_pd = nltk.ELEProbDist(wordtag_uni_fd)
        self.wordtag_bi_cpd = nltk.ConditionalProbDist(wordtag_bi_cfd, nltk.ELEProbDist)
        self.wordtag_tri_cpd = nltk.ConditionalProbDist(wordtag_tri_cfd, nltk.ELEProbDist)
        
        print('Done!')
        
    def pos_tagged_linear_interpolation(self, w2, t2, w1, t1):
        
        alpha=0.6
        beta=0.25
        gamma=0.15
        
        score_1 = -1
        score_2 = -1
        score_3 = -1

        word_1 = 'none'
        word_2 = 'none'
        word_3 = 'none'

        for (w0, t0) in self.wordtag_uni_pd.samples():
            try:
                tri = self.wordtag_tri_cpd[w2, t2, w1, t1].prob(w0)
            except:
                try:
                    tri = self.tricpd[w2, w1].prob(w0)
                except:
                    tri = 0
            try:
                bi = self.wordtag_bi_cpd[w1, t1].prob(w0)
            except:
                try:
                    bi = self.bi_cpd[w1].prob(w0)
                except:
                    bi = 0
            try:
                uni = self.uni_pd.prob(w0)
            except:
                uni = 0
                
            tmp = alpha * tri + beta * bi + gamma * uni    
            if tmp > score_3:
                if tmp > score_2:
                    if tmp > score_1:
                        word_1 = w0, t0
                        score_1 = tmp
                    else:
                        word_2 = w0, t0
                        score_2 = tmp
                else:
                    word_3 = w0, t0
                    score_3 = tmp           
        
        return word_1, word_2, word_3

    def get_score(self, w2, t2, w1, t1, w0, t0):
        alpha=0.6
        beta=0.25
        gamma=0.15
        
        try:
            tri = self.wordtag_tri_cpd[w2, t2, w1, t1].prob(w0)
        except:
            try:
                tri = self.tricpd[w2, w1].prob(w0)
            except:
                tri = 0
        try:
            bi = self.wordtag_bi_cpd[w1, t1].prob(w0)
        except:
            try:
                bi = self.bi_cpd[w1].prob(w0)
            except:
                bi = 0
        try:
            uni = self.uni_pd.prob(w0)
        except:
            uni = 0
                
        tmp = alpha * tri + beta * bi + gamma * uni    
        
        return tmp
        

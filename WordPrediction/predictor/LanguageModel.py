
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
import re

class LanguageModel:
    
    def __init__(self):
        #self.train()
        #print(self.simple_linear_interpolation('and', 'shook'))
        #print(self.pos_tagged_linear_interpolation('and', 'CC', 'shook', ''))
        print('')
    def train(self):
        print('Training...')
        # 3-gram frequency distributions
        tri_fd = nltk.FreqDist()
        tri_cfd = nltk.ConditionalFreqDist()
        tag_tri_cfd = nltk.ConditionalFreqDist()
        wordtag_tri_cfd = nltk.ConditionalFreqDist()

        # 2-gram frequency distributions
        bi_fd = nltk.FreqDist()
        bi_cfd = nltk.ConditionalFreqDist()
        wordtag_bi_cfd = nltk.ConditionalFreqDist()
    
        # 1-gram frequency distributions
        uni_fd = nltk.FreqDist()
        wordtag_uni_fd= nltk.FreqDist()

        genres = []
        
        genres = ['news']
        #output = open('training_output.txt','w')
        for i, genre in enumerate(genres):
            print(repr(i+1) + '/' + repr(len(genres)))
            corpus = brown.tagged_words(categories = genre)
            size = int(len(corpus) * 0.99)
            corpus = corpus[:size]
            trigrams = nltk.trigrams(corpus)
            bigrams = nltk.bigrams(corpus)
            
            for ((word2, tag2), (word1, tag1), (word0, tag0)) in trigrams:
                #output.write(repr((word2, tag2)) + ' ' + repr((word1, tag1)) + ' ' + repr((word0, tag0)))
                tri_fd[word2, word1, word0] += 1      
                tri_cfd[word2, word1][word0] += 1      
                tag_tri_cfd[tag2, tag1][tag0] += 1
                wordtag_tri_cfd[word2, tag2, word1, tag1][word0] += 1

            for ((word1, tag1),(word0, tag0)) in bigrams:
                bi_fd[word1, word0] += 1      
                bi_cfd[word1][word0] += 1    
                wordtag_bi_cfd[word1, tag1][word0] += 1

            for ((word0, tag0)) in corpus:
                uni_fd[word0] += 1      
                wordtag_uni_fd[word0, tag0] += 1
        #output.close()
        # n-gram probability distributions
        self.tri_cpd = nltk.ConditionalProbDist(tri_cfd, nltk.SimpleGoodTuringProbDist)
        self.tri_pd = nltk.SimpleGoodTuringProbDist(tri_fd)
        
        self.bi_cpd = nltk.ConditionalProbDist(bi_cfd, nltk.SimpleGoodTuringProbDist)
        self.bi_pd = nltk.SimpleGoodTuringProbDist(bi_fd)
        
        self.uni_pd = nltk.SimpleGoodTuringProbDist(uni_fd)
        
        # POS n-gram
        self.tag_tri_cpd = nltk.ConditionalProbDist(tag_tri_cfd, nltk.MLEProbDist)
        self.wordtag_uni_pd = nltk.SimpleGoodTuringProbDist(wordtag_uni_fd)
        self.wordtag_bi_cpd = nltk.ConditionalProbDist(wordtag_bi_cfd, nltk.MLEProbDist)
        self.wordtag_tri_cpd = nltk.ConditionalProbDist(wordtag_tri_cfd, nltk.MLEProbDist)
        #print('Done!')
        
    def simple_linear_interpolation(self, w2, w1):
        alpha=0.6
        beta=0.25
        gamma=0.15

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
    
    def getprob_simple_linear_interpolation(self, w2, w1, w0):
        
        alpha=0.6
        beta=0.25
        gamma=0.15

        try:
            tri = self.tri_cpd[w2, w1].prob(w0)
        except:
            tri = self.tri_pd.prob((w2, w1, w0))

        try:
            bi = self.bi_cpd[w1].prob(w0)
        except:
            bi = self.bi_pd.prob((w1, w0))
                
        uni = self.uni_pd.prob(w0)
        return alpha * tri + beta * bi + gamma * uni
    
    def pos_tagged_linear_interpolation(self, w2, t2, w1, t1):
        alpha=0.6
        beta=0.25
        gamma=0.15
        
        best = 0
        word = 'default'

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
            if (tmp > best):
                word = w0
                best = tmp
        
        return word

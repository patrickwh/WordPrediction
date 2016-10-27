
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
import re

class LanguageModel:
    
    def __init__(self):
        self.train()
        #print(self.simple_linear_interpolation('about', 'fallout'))
        #print(self.pos_tagged_linear_interpolation('about', 'ADP', 'fallout', 'NOUN'))
        
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

        genres = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
        'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
        'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
        'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
        'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
        'shakespeare-macbeth.txt', 'whitman-leaves.txt']

        for genre in genres:
            corpus = gutenberg.words(genre)
            corpus = nltk.pos_tag(corpus)
        
            trigrams = nltk.trigrams(corpus)
            bigrams = nltk.bigrams(corpus)
            
            for ((word2, tag2), (word1, tag1), (word0, tag0)) in trigrams:
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
 
        # n-gram probability distributions
        self.tri_cpd = nltk.ConditionalProbDist(tri_cfd, nltk.ELEProbDist)
        self.tri_pd = nltk.ELEProbDist(tri_fd)
        
        self.bi_cpd = nltk.ConditionalProbDist(bi_cfd, nltk.ELEProbDist)
        self.bi_pd = nltk.ELEProbDist(bi_fd)
        
        self.uni_pd = nltk.ELEProbDist(uni_fd)
        
        # POS n-gram
        self.tag_tri_cpd = nltk.ConditionalProbDist(tag_tri_cfd, nltk.ELEProbDist)
        self.wordtag_uni_pd = nltk.ELEProbDist(wordtag_uni_fd)
        self.wordtag_bi_cpd = nltk.ConditionalProbDist(wordtag_bi_cfd, nltk.ELEProbDist)
        self.wordtag_tri_cpd = nltk.ConditionalProbDist(wordtag_tri_cfd, nltk.ELEProbDist)
        print('Done!')
        
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
                    tri = 0.01
            try:
                bi = self.wordtag_bi_cpd[w1, t1].prob(w0)
            except:
                try:
                    bi = self.bi_cpd[w1].prob(w0)
                except:
                    bi = 0.01
            try:
                uni = self.uni_pd.prob(w0)
            except:
                uni = 0.01
                
            tmp = alpha * tri + beta * bi + gamma * uni    
            if (tmp > best):
                word = w0
                best = tmp
        
        return word

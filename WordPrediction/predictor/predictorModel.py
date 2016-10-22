
import nltk
from nltk.corpus import brown
from nltk.corpus import gutenberg
import re

class PredictorModel:
    
    def __init__(self):
        'void constructor'
        self.initialization()
    
    def initialization(self):   
        'initialize the model'
        corpusName = 'emma'
        genre = 'news'
        corpus = self.getCorpus(corpusName,genre)
        tagged_corpus = self.getCorpus(corpusName,genre,tagged=True)
        
        # initialize the parameters for smoothing techniques
        self.abgset1 = [0.8,0.15,0.05]
        self.abgset2 = [0.6,0.25,0.15]
    
        # n-gram frequency distributions
        self.trigrams = nltk.trigrams(tagged_corpus)
        self.tagedbigram = nltk.bigrams(tagged_corpus)
        self.tricfd = nltk.ConditionalFreqDist()
        self.trifd = nltk.FreqDist()
        self.postrifreq = nltk.ConditionalFreqDist()
        self.wordtagbigramfreq = nltk.ConditionalFreqDist()
        self.wordtagtrigramfreq = nltk.ConditionalFreqDist()
        for ((word2,tag2),(word1,tag1),(word0,tag0)) in self.trigrams:
            self.tricfd[word2,word1][word0] += 1
            self.trifd[(word2,word1,word0)] += 1
            self.postrifreq[tag2,tag1][tag0] += 1
            self.wordtagtrigramfreq[word2,tag2,word1,tag1][word0] += 1
        for ((word1,tag1),(word0,tag0)) in self.tagedbigram:
            self.wordtagbigramfreq[word1,tag1][word0] += 1
        self.bicfd = nltk.ConditionalFreqDist(nltk.bigrams(corpus))
        self.bifd = nltk.FreqDist(nltk.bigrams(corpus))
        self.unifd = nltk.FreqDist(corpus)
        self.taggedFreq= nltk.FreqDist(tagged_corpus)
    
        # n-gram probability distributions
        self.tricpd = nltk.ConditionalProbDist(self.tricfd,nltk.ELEProbDist)
        self.tripd = nltk.ELEProbDist(self.trifd)
        self.bicpd = nltk.ConditionalProbDist(self.bicfd,nltk.ELEProbDist)
        self.bipd = nltk.ELEProbDist(self.bifd)
        self.unipd = nltk.ELEProbDist(self.unifd)
        self.unitaggedprob = nltk.ELEProbDist(self.taggedFreq)
    
        # POS n-gram
        self.postriprob = nltk.ConditionalProbDist(self.postrifreq,nltk.ELEProbDist)
        self.wordtagbiprob = nltk.ConditionalProbDist(self.wordtagbigramfreq,nltk.ELEProbDist)
        self.wordtagtriiprob = nltk.ConditionalProbDist(self.wordtagtrigramfreq,nltk.ELEProbDist)
        
    def getCorpus(self,corpus,genre='',tagged=False):
        if corpus=='brown':
            if tagged:
                return brown.tagged_words(categories=genre,tagset='universal')
            else:
                return brown.words(categories=genre)
        if corpus=='emma':
            words = gutenberg.words('austen-emma.txt')
            if tagged:
                return self.tokensFromFile(corpus+"_tag"+".dat")
            else:
                return words
    
    def getLinearScore(self,w2,w1,w0,alpha=0.6,beta=0.25,gamma=0.15):
        try:
            tri = self.tricpd[w2,w1].prob(w0)
        except:
            tri = self.tripd.prob((w2,w1,w0))
        try:
            bi = self.bicpd[w1].prob(w0)
        except:
            bi = self.bipd.prob((w1,w0))
        tmp = alpha*tri + beta*bi + gamma*self.unipd.prob(w0)
        return tmp
    
    def getWordTagScore(self,w2,t2,w1,t1,w0,t0,alpha=0.6,beta=0.25,gamma=0.15):
        try:
            tri = self.wordtagtriiprob[w2,t2,w1,t1].prob(w0)
        except:
            try:
                tri = self.tricpd[w2,w1].prob(w0)
            except:
                tri = 0
        try:
            bi = self.wordtagbiprob[w1,t1].prob(w0)
          
        except:
            try:
                bi = self.bicpd[w1].prob(w0)  
            except:
                bi=0
        try:
            uni = self.unipd.prob(w0)
        except:
            uni = 0
        tmp = alpha*tri + beta*bi + gamma*uni
        return tmp
    
    def linearGuess(self,alpha=0.6,beta=0.25,gamma=0.15):
        word = ''
        best = 0
        for x in self.unipd.samples():
            if(x.startswith(self.chars)):
                tmp = self.getLinearScore(self.word2, self.word1, x, alpha, beta, gamma)
                if(tmp>best):
                    best = tmp
                    word = x
        return word,best
    
    
    def posGuess(self,type=2,a=1):
        b = 1-a
        best = 0
        guess = ''
        if type==1 :
            for (word,tag) in self.taggedFreq:
                if(word.startswith(self.chars)):
                    try:
                        tagscore = self.postriprob[self.tag2,self.tag1].prob(tag)
                    except:
                        tagscore = 0
                    linear = self.getLinearScore(self.word2, self.word1, word)
                    tmp = a*linear + b*tagscore
                    if(tmp>best):
                        best = tmp
                        guess = word
                        bt = tag
            print('ts ',self.postriprob[self.tag2,self.tag1].prob(bt));
            print('ws ',self.getLinearScore(self.word2, self.word1, guess));
        elif type==2 :
            for (word,tag) in self.taggedFreq:
                if(word.startswith(self.chars)):
                    try:
                        tagscore = self.postriprob[self.tag2,self.tag1].prob(tag)
                    except:
                        tagscore = 0
                    wordtagscore = self.getWordTagScore(self.word2,self.tag2, self.word1,self.tag1, word, tag)
                    tmp = a*wordtagscore + b*tagscore
                    if(tmp>best):
                        best = tmp
                        guess = word
        return guess
    
    
    def backoffGuess(self):
        guess = ''
        try:
            guess = self.getFreqGuess(self.tricfd[self.word2,self.word1])
        except ValueError:
            try:
                guess = self.getFreqGuess(self.bicfd[self.word1])
            except ValueError:
                try:
                    guess = self.getFreqGuess(self.unipd)
                except ValueError:
                    print("Backoff No matches")
        return guess
    
    
    def toFile(self,obj, fileName):
        output = open(fileName,'w')
        for x in obj:
            output.write(str(x)+" ")
        output.close()
    
    
    def tokensFromFile(self,file):
        inputm = open(file,'r')
        reg1 = "'[A-Za-z,.]+', '[A-Za-z,.]+'"
        reg2 = "[A-Za-z,.]+"
        tups = re.findall(reg1,inputm.read())
        listm =[]
        for tup in tups:
            x = re.findall(reg2,tup)
            listm.append((x[0],x[2]))
        return listm    
    
    def getFreqGuess(self,freqd):
        guess = None
        if(self.chars==""):
            guess = freqd.max()
        else:
            for w in freqd.most_common(len(freqd)):
                if w[0].startswith(self.chars):
                    guess = w[0];
        if guess==None:
            raise ValueError
        return guess
    
    def getPrediction(self,inputStr):

        t = nltk.pos_tag(nltk.word_tokenize(inputStr),'universal')
        l = len(t)
        # print(t)
    
        if(inputStr.endswith(" ")):
            self.word2 = t[l-2][0]
            self.word1 = t[l-1][0]
            self.tag2  = t[l-2][1]
            self.tag1  = t[l-1][1]
            self.chars = ""
        else:
            self.word2 = t[l-3][0]
            self.word1 = t[l-2][0]
            self.tag2  = t[l-3][1]
            self.tag1  = t[l-2][1]
            self.chars = t[l-1][0]
    
        print(inputStr)
        backoff=self.backoffGuess()   
        linear=self.linearGuess()[0]
        posguess=self.posGuess()
        return (backoff,linear,posguess)
    
    def testModel(self,type,filename='test'):
        'test the accuracy of the model'
        try:
            fp = open(filename+'.txt','r')
            testStrng = fp.read()
            fp.close()
            tokens = nltk.word_tokenize(testStrng)
            tlen = len(tokens)
            
        except Exception:
            print('ERROR')

        if type=='linear_parameter':
            alphas = 0.6
            alphasp = 0.05
            alphae = 0.9
            alphan = (alphae-alphas)/alphasp
            for ai in range(int(alphan)+1):
                alpha = alphas+ai*alphasp
                betas = 0.05
                betasp = 0.05
                betae = 0.95 - alpha
                betan = (betae-betas)/betasp
                for bi in range(int(betan+0.5)+1):
                    beta = betas+bi*betasp
                    gamma = 1-alpha-beta
                    sum = 0
                    for i in range(tlen-2):
                        self.word2 = tokens[i]
                        self.word1 = tokens[i+1]
                        self.chars = ""
                        predict = self.linearGuess(alpha,beta,gamma)[0]
                        if (predict == tokens[i+2]):
                            # print(i,' ',self.word1,' ',self.word2,' ',predict)
                            sum += 1
                    print(alpha,'' ,beta,' ',gamma,' ',sum)
                    print('total number: ',sum,' precision: ',sum/(tlen-2))
        
        elif type=='compare':
            lsum = 0
            bsum = 0
            psum = 0
            for i in range(tlen-2):
                self.word2 = tokens[i]
                self.word1 = tokens[i+1]
                t = nltk.pos_tag((self.word2,self.word1),'universal')
                self.tag2 = t[0][1]
                self.tag1 = t[1][1]
                self.chars = ""
                lp = self.linearGuess()[0]
                bp = self.backoffGuess()
                pp = self.posGuess()
                correct = False
                if  lp == tokens[i+2]:
                    correct = True
                    lsum += 1
                if bp == tokens[i+2]:
                    correct = True
                    bsum += 1
                if pp == tokens[i+2]:
                    correct = True
                    psum += 1
                if correct:
                    print(self.word2,' ',self.word1,' ',lp,' ',bp,' ',pp,' ',tokens[i+2])
            print('linear: ',lsum,' ',(lsum/(tlen-2)),'backoff: ',bsum,' ',(bsum/(tlen-2)),\
                  'pos: ',psum,' ',(psum/(tlen-2)))
        
        
        
        

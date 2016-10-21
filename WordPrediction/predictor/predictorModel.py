
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
    
        # n-gram frequency distributions
        self.trigrams = nltk.trigrams(tagged_corpus)
        self.tricfd = nltk.ConditionalFreqDist()
        self.trifd = nltk.FreqDist()
        self.postrifreq = nltk.ConditionalFreqDist()
        for ((word2,tag2),(word1,tag1),(word0,tag0)) in self.trigrams:
            self.tricfd[word2,word1][word0] += 1
            self.trifd[(word2,word1,word0)] += 1
            self.postrifreq[tag2,tag1][tag0] += 1
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
    
        # POS n-gram
        self.postriprob = nltk.ConditionalProbDist(self.postrifreq,nltk.ELEProbDist)
        
    def getCorpus(self,corpus,genre='',tagged=False):
        if corpus=='brown':
            if tagged:
                return brown.tagged_words(categories=genre,tagset='universal')
            else:
                return brown.words(categories=genre)
        if corpus=='emma':
            words = gutenberg.words('AllText.txt')
            if tagged:
                return self.tokensFromFile(corpus+"_tag"+".dat")
            else:
                return words


    def linearGuess(self,alpha=0.8,beta=0.15,gamma=0.05):
        word = ''
        best = 0
        for x in self.unipd.samples():
            if(x.startswith(self.chars)):
                try:
                    tri = self.tricpd[self.word2,self.word1].prob(x)
                except:
                    tri = self.tripd.prob((self.word2,self.word1,x))
                try:
                    bi = self.bicpd[self.word1].prob(x)
                except:
                    bi = self.bipd.prob((self.word2,x))
                tmp = alpha*tri + beta*bi + gamma*self.unipd.prob(x)
                if(tmp>best):
                    best = tmp
                    word = x
        return word
    
    
    def posGuess(self):
        alpha = 0.5
        beta  = 0.5
        best = 0
        guess = ''
        for (word,tag) in self.taggedFreq:
            if(word.startswith(self.chars)):
                try:
                    bi = self.bicpd[self.word1].prob(word)
                except:
                    bi = self.bipd.prob((self.word1,word))
                try:
                    tri = self.postriprob[self.tag2,self.tag1].prob(tag)
                except:
                    tri = 0
                tmp = alpha*bi + beta*tri
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
                    guess = self.getFreqGuess(self.unifd)
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
        print(t)
    
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
        linear=self.linearGuess()
        posguess=self.posGuess()
        return (backoff,linear,posguess)
    
    def testModel(self,type):
        'test the accuracy of the model'
        try:
            fp = open('test.txt','r')
            testStrng = fp.read()
            fp.close()
            tokens = nltk.word_tokenize(testStrng)
            tlen = len(tokens)
            
        except Exception:
            print('ERROR')

        if type=='linear':
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
                        self.word1 = tokens[i]
                        self.word2 = tokens[i+1]
                        self.chars = ""
                        predict = self.linearGuess(alpha,beta,gamma)
                        if (predict == tokens[i+2]):
                            print(i,' ',self.word1,' ',self.word2,' ',predict)
                            sum += 1
                    print(alpha,'' ,beta,' ',gamma,' ',sum)
                    print(sum/(tlen-2))
        
        elif type=='test':
            lsum = 0
            bsum = 0
            psum = 0
            for i in range(tlen-2):
                self.word1 = tokens[i]
                self.word2 = tokens[i+1]
                self.chars = ""
                lp = self.linearGuess()
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
                    print(self.word1,' ',self.word2,' ',lp,' ',bp,' ',pp,' ',tokens[i+2])
            print('linear: ',lsum,' ',(lsum/(tlen-2)),'backoff: ',bsum,' ',(bsum/(tlen-2)),\
                  'pos: ',psum,' ',(psum/(tlen-2)))
        
        
        
        

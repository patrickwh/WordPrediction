import nltk
from nltk.corpus import brown

class Tester:

    def __init__(self):
        print('')
    
    def run(self, model):
        print('Testing...')
        
        score_simple_linear = 0
        score_pos_tagged_linear = 0
        
        genres = ['news']
        output = open('testing_output.txt','w')
        for i, genre in enumerate(genres):
            print(repr(i+1) + '/' + repr(len(genres)))
            corpus = brown.tagged_words(categories = genre)
            size = int(len(corpus) * 0.99)
            corpus = corpus[size:]
            trigrams = nltk.trigrams(corpus)
            
            word = ''
            total = 0

            for ((word2, tag2), (word1, tag1), (word0, tag0)) in trigrams:
                output.write(repr((word2, tag2)) + ' ' + repr((word1, tag1)) + ' ' + repr((word0, tag0)))
                total += 1
                if(model.simple_linear_interpolation(word2, word1) == word0):
                    score_simple_linear += 1

                if(model.pos_tagged_linear_interpolation(word2, tag2, word1, tag1) == word0):
                    score_pos_tagged_linear += 1
        output.close()
        print(score_simple_linear / total)
        print(score_pos_tagged_linear / total)

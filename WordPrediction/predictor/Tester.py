import nltk

class Tester:

    def __init__(self):
        self.setup()

    def setup(self):
        fp = open('test_corpus.txt', 'r')
        raw = fp.read()
        fp.close()
        tokens = nltk.word_tokenize(raw)
        tokens = nltk.pos_tag(tokens)
        self.trigrams = nltk.trigrams(tokens)
    
    def run(self, model):
        print('Testing...')

        word = ''
        total = 0
        score_simple_linear = 0
        score_pos_tagged_linear = 0
        for ((word2, tag2), (word1, tag1), (word0, tag0)) in self.trigrams:
            total += 1
            if(model.simple_linear_interpolation(word2, word1) == word0):
                score_simple_linear += 1

            if(model.pos_tagged_linear_interpolation(word2, tag2, word1, tag1) == word0):
                score_pos_tagged_linear += 1

        print(score_simple_linear / total)
        print(score_pos_tagged_linear / total)

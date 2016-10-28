import nltk
import math
from nltk.corpus import brown

class Tester:

    def __init__(self):
        print('')
    
    def run(self, model):
        print('Testing...')

        perplexity = 1

        genres = ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies',
            'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance',
            'science_fiction']
        total = 0
        for i, genre in enumerate(genres):
            print(repr(i+1) + '/' + repr(len(genres)))
            corpus = brown.tagged_words(categories = genre)
            size = int(len(corpus) * 0.90)
            corpus = corpus[size:]
            trigrams = nltk.trigrams(corpus)

            for ((word2, tag2), (word1, tag1), (word0, tag0)) in trigrams:
                total += 1
                score = model.get_score(word2, tag2, word1, tag1, word0, tag0)
                perplexity += math.log(score, 2)
        perplexity = perplexity / total
        perplexity = math.pow(2, -perplexity)
        print(perplexity)        

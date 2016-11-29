import nltk
from LanguageModelA import LanguageModelA
from LanguageModelB import LanguageModelB
from Tester import Tester

if __name__=='__main__':
    model_a = LanguageModelA()
    model_b = LanguageModelB()

    arg = 0
    print('Enter 1 to quit')
    print('Enter at least 2 words')
    while arg != 1:
        print(' ')
        arg = input(':_ ')
        text = nltk.word_tokenize(arg)
        if(text[0] != 1):
            text = nltk.pos_tag(text)
            print('Entered ' + repr(text))
            print(' ')
            print('Guessing next word...')
            size = len(text)
            print('POS tagged model said: ' + repr(model_a.pos_tagged_linear_interpolation(
                text[size-2][0], text[size-2][1], text[size-1][0], text[size-1][1])))
            print('Simple LI model said: ' + repr(model_b.simple_linear_interpolation(text[size-2][0], text[size-1][0]))) # get the word from list.
    print('Good bye...')
        
    #tester = Tester()
    #print('Lang model A')
    #tester.run(model_a)

    #print('Lang model B')
    #tester.run(model_b)

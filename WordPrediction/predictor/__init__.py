from LanguageModelA import LanguageModelA
from LanguageModelB import LanguageModelB
from Tester import Tester

if __name__=='__main__':
    #model_a = LanguageModelA()
    model_b = LanguageModelB()
    
    tester = Tester()
    #print('Lang model A')
    #tester.run(model_a)

    print('Lang model B')
    tester.run(model_b)

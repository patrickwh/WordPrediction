from LanguageModel import LanguageModel
from Tester import Tester

if __name__=='__main__':
    model = LanguageModel()
    tester = Tester()
    tester.run(model)

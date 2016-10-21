from predictor.predictorGUI import PredictorGUI
from predictor.predictorModel import PredictorModel

if __name__=='__main__':
    model = PredictorModel()
    model.testModel('test')
    gui = PredictorGUI(model)
    gui.show()
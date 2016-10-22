from predictor.predictorGUI import PredictorGUI
from predictor.predictorModel import PredictorModel

if __name__=='__main__':
    model = PredictorModel()
    model.testModel('compare','test4')
    # model.testModel('linear_parameter','test4')
    gui = PredictorGUI(model)
    gui.show()
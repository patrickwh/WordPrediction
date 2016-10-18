

from  tkinter import *

class PredictorGUI:
    
    def onPredict(self):
        'call back function of the predict button'
        inputStr = self.acceptSentence.get()
        prediction=self.model.getPrediction(inputStr)
        self.backoffDis.delete(0, END)
        self.backoffDis.insert(END, prediction[1])
        self.linearDis.delete(0, END)
        self.linearDis.insert(END, prediction[2])
        self.posDis.delete(0, END)
        self.posDis.insert(END, prediction[3])
       
    def __init__(self,model):
        self.model = model
        
    def show(self):
        top=Tk()
        top.geometry('430x160')
        
        titleHeight=2
        
        titleFrame=Frame(top)
        titleFrame.pack(fill=X)
        self.titleLabel=Label(titleFrame,text="WORD PREDICTOR",font=('Dotum',24,'bold'),\
                    bg='#BFEFFF',height=titleHeight)
        self.titleLabel.pack(expand=1,fill=X)
        
        self.inputFrame=Frame(top)
        self.inputFrame.pack(fill=X,side=TOP)
        self.inputLabel=Label(self.inputFrame,text="Input",font=('Aharoni',12))
        self.inputLabel.pack(side=LEFT)
        self.acceptSentence=Entry(self.inputFrame,textvariable=E)
        self.acceptSentence.pack(expand=1,fill=X,side=LEFT)
        self.predicteButton=Button(self.inputFrame,text="predict",command=self.onPredict)
        self.predicteButton.pack(side=LEFT)
        
        displayFrame1=Frame(top)
        displayFrame1.pack(side=LEFT)
        
        self.displayLabel1=Label(displayFrame1,text="BackOff")
        self.displayLabel1.pack()
        self.backoffDis=Entry(displayFrame1)
        self.backoffDis.pack()
        
        displayFrame2=Frame(top)
        displayFrame2.pack(side=LEFT)
        
        self.displayLabel2=Label(displayFrame2,text="Linear")
        self.displayLabel2.pack()
        self.linearDis=Entry(displayFrame2)
        self.linearDis.pack()
        
        displayFrame3=Frame(top)
        displayFrame3.pack(side=LEFT)
        
        self.displayLabel3=Label(displayFrame3,text="posGuess")
        self.displayLabel3.pack()
        self.posDis=Entry(displayFrame3)
        self.posDis.pack()
        top.mainloop()

#pip install pyqt5designer
#pip install pyqt5 tools 
#pyuic5 Gui.ui -o Gui.py

import pickle
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QFileDialog,QDialog
from PyQt5 import QtCore
from Gui import Ui_Dialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from pandas_datareader import DataReader
model = load_model("model_invest_LTSM2.h5")
Qt = QtCore.Qt



def GetData_Predict():  
    data = DataReader('AAPL', data_source='yahoo', start='2021-01-11', end='2022-06-15')

    return data

def GetData_Show():  
   
    data = DataReader('AAPL', data_source='yahoo', start='2010-01-01', end='2021-01-11')

    return data


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return QtCore.QVariant(str(
                    self._data.iloc[index.row()][index.column()]))
        return QtCore.QVariant()
    
    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None


class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.main_win = QMainWindow()
        self.uic = Ui_Dialog()
        self.uic.setupUi(self.main_win)
        self.uic.btnBieuDo.clicked.connect(self.BieuDo)
        self.uic.btnDuDoan.clicked.connect(self.DuDoan)
        self.uic.btnCapNhat.clicked.connect(self.CapNhat)
        self.CapNhat()      
        
    def CapNhat(self):
        model = PandasModel(GetData_Show().iloc[::-1].round(decimals=3))
        self.uic.tableView.setModel(model)
        
    def BieuDo(self):
        df = GetData_Show()       
        plt.plot(df['Close'], color="green",label=f"Price")
        plt.title(f"Share price")
        plt.xlabel("Time")
        plt.ylabel(f"Share price")
        plt.legend()
        plt.show()
        
    def DuDoan(self):
        df = GetData_Predict()
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit_transform(df['Close'].values.reshape(-1,1))
        
        total_dataset = df['Close']
        model_input = total_dataset[len(total_dataset) - 60:].values
        model_input = model_input.reshape(-1,1)
        model_input = scaler.transform(model_input)
        
        real_data = [model_input[len(model_input) - 60:len(model_input+1),0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

        predict = model.predict(real_data)
        predict = scaler.inverse_transform(predict)
        
        if(GetData_Show().tail(1)['Close'].values[0] > predict[0][0]):
            Message(str(GetData_Show().tail(1)['Close'].values[0]) + " giảm xuống " + str(predict[0][0]))
        else: 
            Message(str(GetData_Show().tail(1)['Close'].values[0]) + " tăng lên " + str(predict[0][0]))
    
    def show(self):
        self.main_win.show()
        
def Message(Text):
    msg=QMessageBox()
    msg.setText(Text)
    msg.exec_()
    
if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show() 
    app.exec_() 
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(838, 448)
        self.btnDuDoan = QtWidgets.QPushButton(Dialog)
        self.btnDuDoan.setGeometry(QtCore.QRect(140, 330, 131, 61))
        self.btnDuDoan.setObjectName("btnDuDoan")
        self.tableView = QtWidgets.QTableView(Dialog)
        self.tableView.setGeometry(QtCore.QRect(60, 40, 721, 261))
        self.tableView.setObjectName("tableView")
        self.btnBieuDo = QtWidgets.QPushButton(Dialog)
        self.btnBieuDo.setGeometry(QtCore.QRect(350, 330, 131, 61))
        self.btnBieuDo.setObjectName("btnBieuDo")
        self.btnCapNhat = QtWidgets.QPushButton(Dialog)
        self.btnCapNhat.setGeometry(QtCore.QRect(550, 330, 131, 61))
        self.btnCapNhat.setObjectName("btnCapNhat")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btnDuDoan.setText(_translate("Dialog", "Dự đoán"))
        self.btnBieuDo.setText(_translate("Dialog", "Biểu đồ"))
        self.btnCapNhat.setText(_translate("Dialog", "Cập nhật"))
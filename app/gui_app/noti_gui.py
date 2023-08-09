# -*- coding: utf-8 -*-

# Author
#
# Dang BH
#
# KSE Company

# Form implementation generated from reading ui file 'ui_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Noti_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.setEnabled(True)
        # Form.resize(780, 466)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setAutoFillBackground(True)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")


        font = QtGui.QFont('Times New Roman',20)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        
    
        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)
        self.horizontalLayout.addLayout(self.verticalLayout)
        
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        font_lb = QtGui.QFont('Times New Roman',35)
        font_lb.setBold(True)
        font_lb.setItalic(True)
        font_lb.setWeight(75)

        self.label = QtWidgets.QLabel(Form)
        self.label.setFont(font_lb)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setFont(font_lb)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setFont(font_lb)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setFont(font_lb)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)


        #--------------buttons-----------------------
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        font = QtGui.QFont('Times New Roman',20)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        #chup anh button
        self.chamlai_bt = QtWidgets.QPushButton(Form)
        self.chamlai_bt.setFont(font)
        self.chamlai_bt.setObjectName("chamlai_bt")
        #logout button
        self.dongy_bt = QtWidgets.QPushButton(Form)
        self.dongy_bt.setFont(font)
        self.dongy_bt.setObjectName("dongy_bt")
        

        self.horizontalLayout_2.addWidget(self.chamlai_bt)
        self.horizontalLayout_2.addWidget(self.dongy_bt)
        

        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
    def resize_image(self):
        screenShape = QtWidgets.QDesktopWidget().screenGeometry()
        w=screenShape.width()
        h=screenShape.height()
        self.image_label.resize(w/2,h/2)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Noti"))
        Form.showMaximized()

        self.image_label.setText(_translate("Form", "Camera"))

        
        # self.label.setText(_translate("Form", "Họ và tên: "))
        # self.label_2.setText(_translate("Form", "ID: "))
        # # self.label_4.setText(_translate("Form", "Độ chính xác: "))
        # self.label_3.setText(_translate("Form", "Thời gian điểm danh: "))

        self.chamlai_bt.setText(_translate("Form", "Chấm công lại "))
        self.dongy_bt.setText(_translate("Form", "Đồng ý "))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('Fusion')
    Form = QtWidgets.QWidget()
    ui = Noti_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())


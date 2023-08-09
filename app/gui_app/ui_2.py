# -*- coding: utf-8 -*-

# Author
#
# Dang BH
#
# KSE Company

# Form implementation generated from reading ui file 'ui_main_window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#x
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class Ui_Form(object):
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

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.hzTimeLayout = QtWidgets.QHBoxLayout(Form)
        self.hzTimeLayout.setObjectName("hzTimeLayout")

        font = QtGui.QFont('Times New Roman',16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)

        self.timeCheck = QtWidgets.QLabel(Form)
        self.timeCheck_2 = QtWidgets.QLabel(Form)
        sizePolicy1 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.timeCheck.sizePolicy().hasHeightForWidth())
        self.timeCheck.setSizePolicy(sizePolicy1)
        self.timeCheck.setFont(font)
        self.timeCheck.setObjectName("timeCheck")

        self.timeCheck_2.setSizePolicy(sizePolicy1)
        self.timeCheck_2.setFont(font)
        self.timeCheck_2.setObjectName("timeCheck_2")
        
        # self.timeCheck.setText(f"<html><head/><body><p align=\"center\">Sang</p><p align=\"center\">Check In: asdfasdfasdfasf</p><p align=\"center\">Check Out: asdfasdfasdfsadf</p></body></html>")
        # self.timeCheck_2.setText(f"<html><head/><body><p align=\"center\">Chieu</p><p align=\"center\">Check In: asdfasdfasdfasf</p><p align=\"center\">Check Out: asdfasdfasdfsadf</p></body></html>")

        self.timeCheck.setAlignment(Qt.AlignCenter)
        self.timeCheck_2.setAlignment(Qt.AlignCenter)

        
        self.hzTimeLayout.addWidget(self.timeCheck)
        self.hzTimeLayout.addWidget(self.timeCheck_2)

        self.gridLayout.addLayout(self.hzTimeLayout,0,0,1,1,Qt.AlignHCenter)

        # self.gridLayout.addWidget(self.timeCheck,1,0,1,1,Qt.AlignHCenter)
        # self.gridLayout.addWidget(self.timeCheck_2,1,0,1,2,Qt.AlignHCenter)

        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        # self.image_label.setCursor(QtGui.QCursor(Qt.ArrowCursor))
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("image_label")
        # self.image_label.resize(845,500)
        self.gridLayout.addWidget(self.image_label,1,0,1,1,Qt.AlignHCenter)
        
        font = QtGui.QFont('Times New Roman',20)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)

        #--------------buttons-----------------------

        self.login_bt = QtWidgets.QPushButton(Form)
        self.login_bt.setFont(font)
        self.login_bt.setObjectName("login_bt")
        self.gridLayout.addWidget(self.login_bt,4,0,1,1)
        #chup anh button
        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setFont(font)
        self.control_bt.setObjectName("control_bt")
        #reset DB button
        self.rsDB_bt = QtWidgets.QPushButton(Form)
        self.rsDB_bt.setFont(font)
        self.rsDB_bt.setObjectName("resetDB_bt")
        #logout button
        self.logout_bt = QtWidgets.QPushButton(Form)
        self.logout_bt.setFont(font)
        self.logout_bt.setObjectName("logout_bt")


        self.gridLayout.addWidget(self.control_bt)
        self.gridLayout.addWidget(self.rsDB_bt)
        self.gridLayout.addWidget(self.logout_bt)

        # self.listWidget = QtWidgets.QListWidget(Form)
        # font = QtGui.QFont('Times New Roman',25)
        # font.setBold(True)
        # font.setWeight(75)
        # self.listWidget.setFont(font)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.listWidget.sizePolicy().hasHeightForWidth())
        # self.listWidget.setSizePolicy(sizePolicy)
        # self.listWidget.setObjectName("listWidget")
        # self.verticalLayout_2.addWidget(self.listWidget)
        # self.horizontalLayout.addLayout(self.verticalLayout_2)
       
        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Cam view"))
        Form.showMaximized()
        self.image_label.setText(_translate("Form", "Camera"))
        self.login_bt.setText(_translate("Form", "Đăng Nhập"))
        self.control_bt.setText(_translate("Form", "Chụp ảnh từ camera"))
        self.rsDB_bt.setText(_translate("Form", "Cập nhật lại Database"))
        self.logout_bt.setText(_translate("Form", "Đăng xuất"))
        
        self.control_bt.hide()
        self.logout_bt.hide()
        self.rsDB_bt.hide()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle('Fusion')
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())


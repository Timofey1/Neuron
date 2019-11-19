import numpy
from keras.preprocessing import image
from keras.models import load_model
import sys
from interface2 import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon, QPixmap


class MyWin(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.MyFunction)
        self.ui.pushButton_2.clicked.connect(self.ImageShow)

    def ImageShow(self):
        im_path = self.ui.textEdit.toPlainText()
        pixmap = QPixmap(im_path)
        self.ui.label_3.setPixmap(pixmap)

    def MyFunction(self):
        self.ui.textEdit_2.setText("")
        im_path = self.ui.textEdit.toPlainText()
        img = image.load_img(im_path, target_size=(28, 28), color_mode='grayscale')
        x = image.img_to_array(img)

        x = 255 - x
        x /= 255
        x = x.reshape(784)
        x = numpy.expand_dims(x, axis=0)
        myModel = load_model('SGD25.h5')
        prediction = myModel.predict(x)
        # print(prediction)
        # print(numpy.argmax(myModel.predict(x)))
        self.ui.textEdit_2.setText(str(numpy.argmax(myModel.predict(x))))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_())

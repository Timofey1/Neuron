import numpy
from keras.preprocessing import image
from keras.models import load_model

im_path = "test.png"
img = image.load_img(im_path, target_size=(28, 28), color_mode='grayscale')
x = image.img_to_array(img)

x = 255 - x
x /= 255
x = x.reshape(784)
x = numpy.expand_dims(x, axis=0)
myModel = load_model('SGD25.h5')
prediction = myModel.predict(x)
print(prediction)
print(numpy.argmax(myModel.predict(x)))


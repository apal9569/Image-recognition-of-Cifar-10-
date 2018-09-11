import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

(traind,trainl),(testd,testl)=cifar10.load_data()
trainl = keras.utils.to_categorical(trainl, 10)
testl = keras.utils.to_categorical(testl, 10)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=traind.shape[1:]))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.10))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.10))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

traind = traind.astype('float32')
testd = testd.astype('float32')
traind /= 255
testd /= 255

model.fit(traind,trainl,batch_size=256,epochs=15)

test_loss, test_acc=model.evaluate(testd,testl,verbose=1)
print('Test Accuracy:',test_acc)

pred=model.predict(testd[1:],batch_size=128)
print('predicted values:',pred)




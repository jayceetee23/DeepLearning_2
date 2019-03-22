from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()
#display the first image in the training data
plt.imshow(test_images[11,:,:],cmap='gray')
plt.title('Ground Truth : {}'.format(test_images[11]))
plt.show()

#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# creating network
# Added additional hidden layer. accuracy ~same
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=20, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


# Plot training loss & validation loss
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, 'b-')
plt.plot(epoch_count, validation_loss, 'r-')

plt.legend(['Training Loss',
            'Validation Loss'], loc='upper right')

plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# Plot training accuracy and validation accuracy
training_acc = history.history['acc']
valid_acc = history.history['val_acc']

plt.plot(epoch_count, training_acc, 'b-')
plt.plot(epoch_count, valid_acc, 'r-')

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Acc',
            'Validation Acc'], loc='upper right')
plt.show()

#Predict 6
print("Inference: ", model.predict_classes(test_data[[11], :]))


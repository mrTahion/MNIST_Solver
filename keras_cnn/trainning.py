import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.utils import np_utils

# Img resolution
img_rows, img_cols = 28, 28
num_classes = 10

# Loading mnist dataset
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# Reshape for CNN
X_train = X_train.reshape(60000, img_rows, img_cols, 1)
X_val = X_val.reshape(10000, img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255

# Lable encoding(one-hot)
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)

# Define model. Conv2D-Conv2d-MaxPool2d-Dropout *2 + Flattern-Dense-Dropout-Dense(final)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print(model.summary())


# Fitting model, using mnist.test dataset for val_data
model.fit(X_train, Y_train, batch_size=256, epochs=48, validation_data = (X_val,Y_val), verbose=1)

# Saving model 2json
model_json = model.to_json()
json_file = open("saved_model.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("saved_model.h5")
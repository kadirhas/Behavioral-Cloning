import csv
import cv2
import numpy as np
import sklearn
from keras.models import Model
import matplotlib.pyplot as plt

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

from sklearn.model_selection import train_test_split

train_lines, validation_lines = train_test_split(lines, test_size=0.2)
	
correction_factor = 0.2
correction = [0, correction_factor, correction_factor*-1]
def generator(lines, batch_size=32):
	num_lines = len(lines)
	while 1:
		np.random.shuffle(lines)
		for offset in range(0,num_lines, batch_size):
			batch_lines = lines[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_line in batch_lines:
				for i in range(3):
					source_path = line[i]
					filename = source_path.split('/')[-1]
					current_path = 'data/IMG/' + filename
					image = cv2.imread(current_path)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					images.append(image)
					measurement = float(line[3])+correction[i]
					measurements.append(measurement)
			augmented_images, augmented_measurements = [], []
			for image,measurement in zip(images,measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(measurement*-1.0)
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)
			
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
# samples are multiplied with 6 with augmentation (left right cameras x3, flip x2)
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_lines)*6, validation_data=validation_generator, nb_val_samples=len(validation_lines)*6, nb_epoch=1)

model.save('model.h5')


# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
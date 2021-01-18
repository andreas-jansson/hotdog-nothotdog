import sys
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import random


######################### importerar testdata ############################

files=glob.glob("/Users/andreas/Desktop/programmering/python/deep learning/hot dog not hot dog/hotdog-nothotdog/train/hotdog/*.jpg")
TrainData = []
index = 0

for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    TrainData.append((imgarray, 1))

files=glob.glob("/Users/andreas/Desktop/programmering/python/deep learning/hot dog not hot dog/hotdog-nothotdog/train/nothotdog/*.jpg")
index = 0
for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    TrainData.append((imgarray, 0))

print("Done.")

random.shuffle(TrainData)


files=glob.glob("/Users/andreas/Desktop/programmering/python/deep learning/hot dog not hot dog/hotdog-nothotdog/test/hotdog/*.jpg")
TestData = []
index = 0

for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    TestData.append((imgarray, 1))

files=glob.glob("/Users/andreas/Desktop/programmering/python/deep learning/hot dog not hot dog/hotdog-nothotdog/test/nothotdog/*.jpg")
index = 0
for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    TestData.append((imgarray, 0))

print("Done.")
random.shuffle(TestData)



print("TrainData ",len(TrainData))
print("TrainData ",len(TestData))

#Läser in data till tuples

(train_images, train_lables) = np.array([item[0] for item in TrainData]), np.array([item[1] for item in TrainData])
(test_images, test_lables) = np.array([item[0] for item in TrainData]), np.array([item[1] for item in TrainData])

#normaliserar data -> svart/vit
train_images = train_images / 255.0
test_images = test_images / 255.0

#Behövs inte i think
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

#plt.figure(figsize=(10, 10))

#classnamn för output
class_names = ['not hot dog', 'hot dog']

#dimensioner av datan
print("shape: ", train_images.shape)



model = keras.Sequential([
    keras.layers.AveragePooling2D((2, 2), 2, input_shape=(128, 128, 3)), #minskar storlek, bild size och färg dim
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.1),              #fixar overfitting
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid') #för enkel output 0-1
])

#model.summary()

# optimerar modellen
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

#Tränar modellen 50 ggr
model.fit(train_images, train_lables, epochs=50, batch_size = 100)

#testar modellen med testdata
test_loss, test_acc = model.evaluate(test_images, test_lables)
print(f'Test Accuracy: {test_acc}')


#Save model
model.save('hotdog_nothotdog_ai.model')

#Reload model
#new_model = tf.keras.models.load_model('hotdog_nothotdog_ai.model')

#new predictions with list of images
#predictions2 = new_model.predict([test_images])


# Visar 25 första gissningarna och svar
predictions = model.predict(test_images)

i = 0
while i < 25:
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    if ((predictions[i] >= 0.5) and (test_lables[i] == 1)):
        success = 'Success!'
        x = 1
    elif((predictions[i] < 0.5) and (test_lables[i] == 0)):
        success = 'Success!'
        x= 0
    elif((predictions[i] >= 0.5) and (test_lables[i] == 0)):
        success = 'Fail!'
        x = 1
    elif((predictions[i] < 0.5) and (test_lables[i] == 1)):
        success = 'Fail!'
        x = 0
    else:
        print('failed?!?')

    plt.xlabel(f'Guess: {class_names[x]}, {success}')
    i = i + 1

plt.show()



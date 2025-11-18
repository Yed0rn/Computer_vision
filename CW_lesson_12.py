import tensorflow as tf

from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image


train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',image_size=(128,128),batch_size=30,label_mode='categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',image_size=(128,128),batch_size=30,label_mode='categorical')

normalisation_layer=layers.Rescaling(1./255)

train_ds=train_ds.map(lambda x,y:(normalisation_layer(x),y))
test_ds=test_ds.map(lambda x,y:(normalisation_layer(x),y))

model = models.Sequential()

model.add(
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3))
)
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(3,activation='softmax'))
#компіляція
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#навчання моделі
history = model.fit(train_ds,epochs=50,validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Правдивість: {test_acc}')
class_name = ["cars","cats","dogs"]

img = image.load_img('images/cat.jpg',target_size=(128,128))
image_arr=image.img_to_array(img)
image_arr=image_arr/255.0
image_arr = np.expand_dims(image_arr,axis=0)
prediction = model.predict(image_arr)
predict_index= np.argmax(prediction[0])

print(f'Імовірність пр класам: {prediction[0]}')
print(f'Модель визначила: {class_name[predict_index]}')

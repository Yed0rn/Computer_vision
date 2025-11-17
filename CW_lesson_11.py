import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Sequential
from tensorflow import keras #розширення до тензорфлоу
from tensorflow.keras import layers #створення шарів в нейронці
from sklearn.preprocessing import LabelEncoder #створює мітки
import matplotlib.pyplot as plt #бібліотека яка будує графіки

#2 працюємо з csv файлами
df = pd.read_csv('data/figures.csv')
#print(df.head())

encoder =LabelEncoder()
df['label_enc']=encoder.fit_transform(df['label'])


X =df[['area','perimeter', 'corners']]
y =df['label_enc']

# створюємо модель
model = keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(8,activation='relu'),
    layers.Dense(8,activation='relu'),
    layers.Dense(8,activation='softmax')
])


# компіляція моделі
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X,y,epochs=300,verbose=0)

plt.plot(history.history['loss'],label='Втрати')
plt.plot(history.history['accuracy'],label='Точність')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title("Процес навчання моделі")
plt.legend()
plt.show()



test = np.array([[25,20,0]])

pred = model.predict(test)
print(f'імовірність кожного класу {pred}')
print(f'Модель визначила {encoder.inverse_transform(np.argmax(pred))}')



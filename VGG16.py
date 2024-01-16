from tensorflow.keras.applications import VGG16
from tensorflow.python.keras.layers  import Activation, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))  # 灰階圖的channel數為1
        image_array[i] = image

    return image_array, image_label

def convert_to_3_channels(img_arrays):
    sample_size, nrows, ncols, c = img_arrays.shape
    img_stack_arrays = np.zeros((sample_size, nrows, ncols, 3))
    for _ in range(sample_size):
        img_stack = np.stack(
            [img_arrays[_][:, :, 0], img_arrays[_][:, :, 0], img_arrays[_][:, :, 0]], axis=-1)
        img_stack_arrays[_] = img_stack / 255.0 # normalization
    return img_stack_arrays

def build_model(preModel=VGG16, num_classes=7):
    # 預先訓練好的模型 -- VGG16
    base_model = tf.keras.applications.VGG16(input_shape=(48,48,3),include_top=False,weights="imagenet")

    # VGG16 原有的層均不重新訓練 freez, 不含後三層(辨識層)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
        
    # 連接自訂層
    model=tf.keras.Sequential()
    model.add(base_model)
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256,kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization(synchronized=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization(synchronized=True))
    model.add(Activation('relu'))
    model.add(Dense(7,activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                loss='sparse_categorical_crossentropy', # no need to do one-hot
                metrics=['accuracy'])

    model.summary()
    print("\n-----------------------------------------")
    return model


df_raw = pd.read_csv("./fer2013.csv")
# 資料切割(訓練、驗證、測試)
df_train0 = df_raw[df_raw['Usage'] == 'Training']
df_val = df_train0.sample(frac=0.3, replace=False,random_state=42)

# 獲取 df_train 不重複的索引
train_indices = df_train0.index.difference(df_val.index)
# 使用這些索引獲取剩餘的資料
df_train = df_train0.loc[train_indices]

df_test = df_raw[df_raw['Usage'] == 'PrivateTest']

X_train, y_train = prepare_data(df_train)
X_val, y_val = prepare_data(df_val)
X_test, y_test = prepare_data(df_test)

print(f"Train data: {len(y_train)}")
print(f"Valid data: {len(y_val)}")
print(f"Test data: {len(y_test)}")
print("----------------------------------------------------")
X_train = convert_to_3_channels(X_train)
X_val = convert_to_3_channels(X_val)
X_test = convert_to_3_channels(X_test)

# y_train_oh = to_categorical(y_train)
# y_val_oh = to_categorical(y_val)
# y_test_oh = to_categorical(y_test)

myModel = build_model()
prob_vgg16 = myModel(X_train[:1]).numpy()
print(prob_vgg16.shape)

epochs = 100
batch_size = 128

from tensorflow.keras.callbacks import ReduceLROnPlateau
LR_function=ReduceLROnPlateau(monitor='val_accuracy',
                             patience=3,
                             # 3 epochs 內acc沒下降就要調整LR
                             verbose=1,
                             factor=0.5,
                             # LR降為0.5
                             min_lr=0.001
                             # 最小 LR 到0.00001就不再下降
                             )

history = myModel.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size,callbacks=[LR_function])
print(history)


import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('CNN Metrices (Accuracy)')
plt.ylabel('% value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.savefig('CNN Metrices(Accuracy).png')
# plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('CNN Metrices(Loss)')
plt.ylabel('% value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.savefig('CNN Metrices(Loss).png')
# plt.show()  


y_pred = myModel.predict(X_test)
scores = myModel.evaluate(X_test,y_test)

print("\n-----------------------------------------")
print("test loss, test acc:",scores)
print("-----------------------------------------")

y_pred_labels = []
for i in y_pred:
    y_pred_labels.append(np.argmax(i))
# y_actual = X_test.classes[X_test.index_array]
    

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred_labels)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('Confusion Metrix.png')
# plt.show()

from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))  # 灰階圖的channel數為1
        image_array[i] = image

    return image_array, image_label


def build_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(48,48,1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(512,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),padding='same',kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))



    model.add(Flatten())
    model.add(Dense(256,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(32,kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(7,kernel_initializer='he_normal'))
    model.add(Activation('softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Summarizing the model architecture and printing it out
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


# y_train_oh = to_categorical(y_train)
# y_val_oh = to_categorical(y_val)
# y_test_oh = to_categorical(y_test)

myModel = build_model()
ouputlayer = myModel(X_train[:1]).numpy()
print(ouputlayer.shape)

epochs = 100
batch_size = 128

from tensorflow.keras.callbacks import ReduceLROnPlateau
LR_function=ReduceLROnPlateau(monitor='val_accuracy',
                             patience=3,
                             # 3 epochs 內acc沒下降就要調整LR
                             verbose=1,
                             factor=0.5,
                             # LR降為0.5
                             min_lr=0.01
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

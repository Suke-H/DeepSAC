from keras.layers import Concatenate, Input, Dense, Conv2D, MaxPooling2D, Lambda, Flatten, BatchNormalization
from keras.layers.core import Dropout
from keras.models import Model
from keras.optimizers import Adam

import os
import time

from MakeDataset import *

def SingleNet(N):
    inputs = Input(shape=(N,))

    d1, d2 = 64, 64
    d3, d4, d5 = 64, 128, 1024
    d6, d7, d8 = 512, 216, 1

    x = Dense(d1, activation='relu')(inputs)
    x = Dense(d2, activation='relu')(x)

    #mid_1 = Model(inputs=inputs, outputs=x)

    inputs2 = x

    x = Dense(d3, activation='relu')(inputs2)
    x = Dense(d4, activation='relu')(x)
    x = Dense(d5, activation='relu')(x)

    #mid_2 = Model(inputs=mid_1.output, outputs=x)

    inputs3 = x

    x = Dense(d6, activation='relu')(inputs3)
    x = Dense(d7, activation='relu')(x)
    x = Dense(d8, activation='relu')(x)

    #mid_3 = Model(inputs=mid_2.output, outputs=x)

    SingleNet = Model(inputs=inputs, outputs=x, name="SingleNet")

    return SingleNet

def DeepSAC(M, N):
    # 入力を(1,N)のM個分岐にする
    inputs = [Input(shape=(N,)) for i in range(M)]
    outputs = []
    
    # 入力をSingleNetに入れて出力をまとめる
    model = SingleNet(N)
    for single_input in inputs:
        x = model(single_input)
        outputs.append(x)
    x = Concatenate()(outputs)

    # スコアをソフトマックスで出力
    x = Dense(M, activation='softmax', name="softmax")(x)

    DeepSAC = Model(inputs=inputs, outputs=x, name="DeepSAC")

    return DeepSAC

def VGG16(m, n):
    # inputs = Input(shape=(224, 224, 3))
    inputs = Input(shape=(m, n, 1))
    # Due to memory limitation, images will resized on-the-fly.
    #x = Lambda(lambda image: tf.image.resize_images(image, (224, 224)))(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)
    flattened = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(flattened)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    # CIFAR10は10クラスなので出力の数が違う
    predictions = Dense(m, activation='softmax', name='predictions')(x)

    VGG16 = Model(inputs=inputs, outputs=predictions, name="VGG16")

    return VGG16

#####################################################################

batch_size = 10
epochs = 10
m, n = 500, 100
N = 100

t1 = time.time()
print("データセット読み込み")

# データセット読み込み
#x_train, x_test, y_train, y_test = MakeDataset(N, m, n)
#
# np.save('data/x_train', x_train)
# np.save('data/x_test', x_test)
# np.save('data/y_train', y_train)
# np.save('data/y_test', y_test)

x_train, x_test, y_train, y_test = np.load('data/x_train.npy'), np.load('data/x_test.npy'), np.load('data/y_train.npy'), np.load('data/y_test.npy')

print(np.any(np.isinf(x_train)), np.any(np.isinf(x_test)), np.any(np.isnan(x_train)), np.any(np.isnan(x_test)))
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# x_train = x_train.transpose(1, 0, 2)
# x_train = [x_train[i] for i in range(m)]
# x_test = x_test.transpose(1, 0, 2)
# x_test = [x_test[i] for i in range(m)]

x_train = x_train.reshape(x_train.shape[0], m, n, 1)
x_test = x_test.reshape(x_test.shape[0], m, n, 1)

t2 = time.time()
print("time:{}s".format(t2-t1))
print("定義")


# モデル定義
# 保存していたモデルがあればロード
# if os.path.exists('models/model.h5'):
#     model = keras.models.load_model('models/model.h5', compile=False)
#     print("モデルあったよ")
#
# else:
#     model = DeepSAC(m, n)
#     print("モデルなかったよ")

model = VGG16(m, n)

#from keras.utils import plot_model
#plot_model(model, to_file='data/model3.png')

optimizer = Adam()
model.compile(loss="categorical_crossentropy",optimizer=optimizer)

t3 = time.time()
print("time:{}s".format(t3-t2))
print("訓練")

# 訓練
# validation_dataを入れるとval_loss, val_accも出してくれる
# lossが下がるがval_lossが上がったら過学習

model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test)
        )

# 評価
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score)

# 推定
score = model.predict(x_test)
print('score:{}'.format(score))
print("y:{}".format(y_test))

# 保存
model.save('models/model.h5', include_optimizer=False)
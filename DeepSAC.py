from keras.layers import Concatenate, Input, Dense
from keras.models import Model

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


model = DeepSAC(3, 1)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='data/model.png')
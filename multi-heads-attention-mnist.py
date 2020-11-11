#coding:utf-8

# This is a clone of multi-heads-attention-mnist.py 
# from <https://github.com/johnsmithm/multi-heads-attention-image-classification/blob/master/multi-heads-attention-mnist.py>
# by Mosnoi Ion.
# And added some comments to understand how it works. 　コメント追加
#     change  variance calculation of class NormL.  　　分散の計算方法を変更
#--------------------------------------------------------------------------------------------------------------------------
# multi-heads-attention-mnist.py:
#   Attention is all you need: A Keras Implementation
#   Using attention to increase image classification accuracy
#   The medium article can be found <https://medium.com/@moshnoi2000/all-you-need-is-attention-computer-vision-edition-dbe7538330a4>.
#--------------------------------------------------------------------------------------------------------------------------
#
#  「Attensionの数字文字の画像認識への応用」
#   tf 2.x だと、動作しないかも。


from keras.layers import Dense, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras import backend as K

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import TensorBoard


# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  keras 2.2.4
#  tensorflow  1.12.0



def MultiHeadsAttModel(l=8*8, d=512, dv=64, dout=512, nv = 8 ):
    # 実際mainから呼ばれている引数は MultiHeadsAttModel(l=6*6, d=64*3 , dv=8*3, dout=32, nv = 8 )
    #  入力は(36=画素6x6,192=64*3)=6912
    #
    #  l is numbers of blocks in the feature map (36=6*6)
    #  d is the dimension of the block  (192=64*3)
    #  nv is the number of projections for each block: heads (8)
    #  dv is the dimension of the linear space the input to be projected (24=8*3)
    #  dout is the output of the block after applying this attention (32)36
    #
    #  Multi-head Attention
    #  Attention とは query によって memory(key,value) から必要な情報を選択的に引っ張ってくること
    #  memory から情報を引っ張ってくるときには、 query は key によって取得する memory を決定し、対応する value を取得します。
    #
    
    v1 = Input(shape = (l, d))   # value(36, 192)
    q1 = Input(shape = (l, d))   # query(36, 192)
    k1 = Input(shape = (l, d))   # key(36, 192)
    
    v2 = Dense(dv*nv, activation = "relu")(v1)   # valueに全結合ニューラルネットワークレイヤーをする。(36,192) -> (36,24*8=192)
    q2 = Dense(dv*nv, activation = "relu")(q1)   # queryに全結合ニューラルネットワークレイヤーをする。(36,192) -> (36,24*8=192)
    k2 = Dense(dv*nv, activation = "relu")(k1)   # keyに全結合ニューラルネットワークレイヤーをする。(36,192) -> (36,24*8=192)
    
    # MultiHeads: query, key, value を8headそれぞれ24個に split してからそれぞれ attention を計算する
    v = Reshape([l, nv, dv])(v2)   # 形を変える　(36,24*8=192) -> (36, 8, 24)
    q = Reshape([l, nv, dv])(q2)   # 形を変える　(36,24*8=192) -> (36, 8, 24)
    k = Reshape([l, nv, dv])(k2)   # 形を変える　(36,24*8=192) -> (36, 8, 24)
    
    # q と k の内積/sqrt(8)=scaled dot-product   (36, 8, 24) * (36, 8, 24) -> (36, 8, 8)
    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[-1,-1]) / np.sqrt(dv), output_shape=(l, nv, nv))([q,k])# l, nv, nv
    
    # softmax を取ることで正規化します  (36, 8, 8) -> (36, 8, 8)
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)
    
    # att と v の内積  重みattに従ってvalue を取得　(batch_size, 36, 8, 8) * (batch_size, 36, 8, 24) -> (batch_size, 36, 8, 24)
    #             K.batch_dot(x[0], x[1],axes=[4,3])　 
    #             tensorflow_backend.pyのなかのdef batch_dot(x, y, axes=None)のソースコードを見ると、
    #             axesが定義されて　x[0],x[1]が同じdimensionで　かつ　dimensionが2以外のときの処理は
    #             if axes is not None:
    #                adj_x = None if axes[0] == ndim(x) - 1 else True   #   4 is not (4-1=3), True
    #                adj_y = True if axes[1] == ndim(y) - 1 else None   #   3 is (4-1=3), True
    #             out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    #　　　　　　であり、axes そのものの値を matmulに渡しているわけではない。
    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[4,3]),  output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)  # 8headそれぞれ24個をもとの形に戻す　(36, 8, 24) -> (36, 192)
    
    # 取得したvalueをqueryに付与
    out = Add()([out, q1])  # out + q1  (36,192) + (36,192)
    
    out = Dense(dout, activation = "relu")(out)  #   (36, 192) -> (36, 32)
    
    return  Model(inputs=[q1,k1,v1], outputs=out)

class NormL(Layer):
    #
    # 独自の学習可能な重みをもつKerasレイヤーを作成
    # 学習可能な重みを持つ独自演算は，自身でレイヤーを実装する必要があります
    #　__Keras 2.0__でのレイヤーの枠組みを参照のこと
    #
    #  汎用のLayerNormalization class を使った方がいいかも。
    #
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # 平均と分散を計算して、ゼロ平均の分散１の系列にする
        # 最後の軸に沿って平均と分散を計算している
        #  out = ln_out * self.a(=weight)  +  self.b(=bias)
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        """
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu)  / (sigma + eps)  # この割り算でlossがnanになってしまい、計算がうまく行かない！
        """
        #　そこで、下記の様にコードを書き換えた。
        variance = K.mean(K.square(x - mu), axis=-1, keepdims=True)
        std = K.sqrt(variance + eps)
        ln_out = (x - mu)  / std
        return ln_out*self.a + self.b
        
        
    def compute_output_shape(self, input_shape):
        return input_shape
    
    

if __name__ == '__main__':   

    nb_classes = 10  # (0から9までの数字の数　digit numebr)
    
    # the data, shuffled and split between tran and test sets
    
    
    # 大文字のXがデータで、小文字のyがラベルを示す
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)
    
    X_train = X_train.reshape(60000, 28,28,1) # 訓練用データの 形を変える　(60000個, 28ｘ28画像,色数1)
    X_test = X_test.reshape(10000, 28,28,1)   # テスト用の 形を変える　(10000個, 28ｘ28画像,色数1)
    #  白黒gray スケールから　0.0～1.0に変換
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)
    
    # ラベルの変換　名称が大文字のYになっている
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    
    # 入力画像サイズ　28ｘ28画像,色数1
    inp = Input(shape = (28,28,1))
    x = Conv2D(32,(2,2),activation='relu', padding='same')(inp)  # (28,28,32)
    x = MaxPooling2D(pool_size=(2, 2))(x)                        # (14,14,32)
    x = Conv2D(64,(2,2),activation='relu')(x)                    # (13,13,64)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)        # (7,7,64)
    x = Conv2D(64*3,(2,2),activation='relu')(x)                  # (6,6,192)
    
    
    # Attensionを使う場合は　True を設定する。　使わない場合は、Falseにする。
    #  (36=画素6x6,192=64*3)=6912　から (6,6,32)=1152へ　データ数を圧縮している。
    if True:
        x = Reshape([6*6,64*3])(x)  # 36,192 =6912
        att = MultiHeadsAttModel(l=6*6, d=64*3 , dv=8*3, dout=32, nv = 8 )
        x = att([x,x,x])   # Self-Attention   input (query) と memory (key, value) すべてが同じものを使う Attention
        x = Reshape([6,6,32])(x) # 6x6x32=1152,  (6,6,32)
        x = NormL()(x)  # 、ゼロ平均と分散１に正規化して、学習可能な重みを使って重み付け  (6,6,32)
    
    
    # 全結合ニューラルネットワークレイヤーで　1152(attension使わないときは6912) -> 256 -> 10 (0から9までの数字の数　digit numebr)
    x = Flatten()(x)   # 1次元の並びに変換する
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    
    # モデルを定義する
    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    
    # Sequentialモデルのコンパイル
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    # callback として　tensorboard　を仕掛ける
    tbCallBack = TensorBoard(log_dir='./Graph/mhatt1', histogram_freq=0, write_graph=True, write_images=True)
    
    # モデルの学習とテストを行う
    model.fit(X_train, Y_train,     # 訓練用のデータ
              batch_size=128,       # バッチサイズ
              epochs=100,           # エポック
              verbose=1,            # 動作中に表示される情報の設定
              validation_data=(X_test, Y_test),  # テスト用のデータ
              callbacks=[tbCallBack]  # callbackの設定
             )


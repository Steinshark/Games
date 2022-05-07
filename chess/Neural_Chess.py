import tensorflow as tf
import numpy
import random
from tensorflow import keras
from keras import layers

class Engine:
    def __init__(self,board):
        self.board = board
        self.parameters = {\
                  #possiblemoves board_position
            1     :    218      +     64       ,\
                  #         number_layers
            2     :              3,\
                  #         number_nodes / layer
            3     :              120,\
        }
        self.build_model(self.parameters[1],self.parameters[2],)
    def build_model(self,inputs,j,k):
        self.model = keras.Sequential(Dense(inputs))

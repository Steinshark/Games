import chess
import random
import time
import math
import sys
import tensorflow as tf 
from tensorflow import keras

class Chess_Game:
    def __init__(self):
        self.board = chess.Board()
        self.moves = list()
        self.game_over = False
        self.players = ['white','black']
        self.counter = 0
    def random_move(self):
        sample_size = self.board.legal_moves.count()
        self.moves = [move for move in iter(self.board.legal_moves)]
        return str(self.moves[random.randint(0,sample_size-1)])
    def best_move(self):
        pass
    def play(self):

        self.board = chess.Board(chess.STARTING_FEN)
        self.game_over = False
        game = '{'
        while not self.game_over:
            move = self.random_move()
            self.board.push_san(move)
            game+=str(move)+" "
            self.check_game()
        res = self.board.outcome()
        game+=','+str(res)+'\n'
        self.file.write(game)
        self.counter += 1

        if (self.counter %100) == 0:
            self.file.close()
            self.file = open(sys.argv[1],'a')
            self.counter = 0
    def check_game(self):
        if self.board.outcome() is None:
            return
        else:
            self.game_over = True
    def write_game(self):
        self.file = open(sys.argv[1],'a')


class QLearning:

    def __init__(self):
        self.pieces = {
            "Bpawn":0,
            "Brook":1,
            "Bbishop":2,
            "Bnight":3,
            "Bqueen":4,
            "Bking":5,
            "Wpawn":6,
            "Wrook":7,
            "Wbishop":8,
            "Wnight":9,
            "Wqueen":10,
            "Wking":11}
        self.squares = [f"{file}{rank}" for file in ['a','b','c','d','e','f','g','h'] for rank in range(1,9)]

        self.model = None 

        #Our input vector is [boolean for piece in square] for each square 
        # size 768  
        self.input_key = [f"{piece}_on_{square}" for piece in self.pieces for square in self.squares]
        
        #Our output vector is [each square to every other square] U [castleTypes]
        # Size 4036
        self.output_key = [f"{fr}_to_{to}" for fr in self.squares for to in self.squares if not fr == to]
        self.output_key += ["Wcastlesk","Wcastlesq","Bcastlesk","Bcastlesq"]
        

        print(f"input: size: {len(self.input_key)}\n{self.input_key}")
        print(f"\n\n\noutput: size: {len(self.output_key)}\n{self.output_key}")

    def build_model(self):
        #Input to the model is represented as:

        # all possible distinct pieces as a dimesnion 
        # BKnight, BRo 
        # [][][]...[]
        pass


        self.model = keras.Sequential()
        self.model.append()



if __name__ == "__main__":
    ql = QLearning()



'''
        self.com_model=tf.keras.Sequential([
            tf.keras.layers.Dense(input_dim,activation='relu'),
            tf.keras.layers.Dense(1024,activation='relu'),          ##POTENTIALLY ONLY HAVE THIS LAYER
            tf.keras.layers.Dense(2,activation='relu'),         ##POTENTIALLY ONLY HAVE THIS LAYER
            tf.keras.layers.Dense(output_dim)
          ])
        self.com_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=tf.keras.losses.MeanSquaredError(),metrics=['MSE'])
        #Train the com model
        t1 = time.time()
        self.com_model.fit(com_data_X,com_data_y,
            validation_split=.25, #auto train/test splitting...
            callbacks=[tf.keras.callbacks.EarlyStopping(restore_best_weights=True,patience=3)],
            epochs=_EPOCHS,
            batch_size=1024,
            verbose=1)
        print(f"trained model in {(time.time()-t1):.3f}s")  
'''
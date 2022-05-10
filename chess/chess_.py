import chess
import random
import time
import math
import sys
import numpy

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Dense 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



class ChessGame:
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

    def get_legal_moves(self):
        return [self.board.lan(move)[-5:] for move in iter(self.board.legal_moves)]

    def get_state_vector(self):
        pieces = {"p":0,"r":1,"b":2,"n":3,"q":4,"k":5,"P":6,"R":7,"B":8,"N":9,"Q":10,"K":11}
        fen = self.board.fen()
        i = 0
        c_i = 0
        board_vect = []
        while i < 64:
            char = fen[c_i]
            square = [0,0,0,0,0,0,0,0,0,0,0,0]
            if char in ["1","2","3","4","5","6","7","8"]:
                for _ in range(int(char)):
                    square = [0,0,0,0,0,0,0,0,0,0,0,0]
                    board_vect += square
                    i += 1
            elif char == " ":
                break
            elif char == "/":
                pass
            else:
                square[pieces[char]] = 1
                board_vect += square
                i += 1

            c_i += 1

        if self.board.turn == chess.WHITE:
            board_vect += [1,0]
        else:
            board_vect += [0,1]

        for color in [chess.WHITE,chess.BLACK]:
            board_vect += [ int(self.board.has_kingside_castling_rights(color)),
                            int(self.board.has_queenside_castling_rights(color))]

        return numpy.transpose(numpy.array(board_vect).shape)

    def get_terminal_reward(self):
        outcome = self.board.outcome()
        if outcome.winner == chess.WHITE:
            return 10 
        elif outcome.winner == chess.BLACK:
            return 10
        elif outcome.termination == Termination.INSIFFICIENT_MATERIAL:
            return -.2
        elif outcome.termination == Termination.SEVENTYFIVE_MOVES:
            return -.2
        else: 
            return 0

    def get_this_move_reward(self,given_move):
        return 1

    def play(self):

            self.board = chess.Board(chess.STARTING_FEN)
            self.game_over = False
            while not self.game_over:
                move = self.random_move()
                self.board.push_san(move)
                self.check_game()
            res = self.board.outcome()
            print(res)

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

        self.rewards = {
            "capture"   : 1,
            "checkmate" : 10,
            "tie"       : .5,
            "losePiece" : -1,
            "getMated"  : -10,
            "check"     : .5
        }

        self.squares = [f"{file}{rank}" for file in ['a','b','c','d','e','f','g','h'] for rank in range(1,9)]
        # Two networks, one to learn, one as the target output 
        self.learning_model = None 
        self.target_model   = None 

        #Our input vector is [boolean for piece in square] for each square 
        # size 768  
        self.input_key = [f"{piece}_on_{square}" for piece in self.pieces for square in self.squares] + ["Wmove","Bmove"]
        self.input_key += ["Wcastlesk","Wcastlesq","Bcastlesk","Bcastlesq"]
        
        #Our output vector is [each square to every other square] U [castleTypes]
        # Size 4036
        self.output_key = [f"{fr}-{to}" for fr in self.squares for to in self.squares if not fr == to]
        self.output_key += [f"{pawn}-{sq}={pi}" for pawn in ["a7","b7","c7","d7","d7","e7","f7","g7"] for sq in ["a8","b8","c8","d8","d8","e8","f8","g8"] for pi in ["Q","R","B","N"]]
        self.output_key += [f"{pawn}-{sq}={pi}" for pawn in ["a2","b2","c2","d2","d2","e2","f2","g2"] for sq in ["a1","b1","c1","d1","d1","e1","f1","g1"] for pi in ["q","r","b","n"]]
        self.output_key += ["OO","OOO"]
        

        print(f"input: size: {len(self.input_key)}")
        print(f"output: size: {len(self.output_key)}")

    def build_model(self):
        self.learning_model = keras.Sequential([
            Dense(len(self.input_key),activation="relu"),
            Dense(8124,activation="relu"),
            Dense(512,activation="relu"),
            Dense(512,activation="relu"),
            Dense(len(self.output_key))])

        self.target_model = keras.Sequential([
            Dense(len(self.input_key),activation="relu"),
            Dense(8124,activation="relu"),
            Dense(512,activation="relu"),
            Dense(512,activation="relu"),
            Dense(len(self.output_key))])

    def train_model(self,iterations):

        # used for experience replay 
        self.experiences = {}

        #Get a new game
        game = ChessGame()
        for i in range(iterations):

            #Make the move according to the model
            given_move = self.target_model.predict(game.get_state_vector())[0][2]
            print(f"move: {given_move}")



            #Get the move's reward
            if not game.board.outcome() is None:
                reward = game.get_terminal_reward()
                print(f"game ended with {reward}")
                game = ChessGame()
            
            else:

                #get reward of newest state 
                reward = game.get_this_move_reward() + self.discount_factor * self.target_model.predict(game.state_vector)
                self.experiences.append(reward)

            #Check for experience 

if __name__ == "__main__":
    model = QLearning()
    model.build_model()
    model.train_model(10)
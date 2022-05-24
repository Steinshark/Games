import torch 
import torch.nn as nn
import torch.nn.functional as F 
import chess
from string import ascii_uppercase
class ChessNetV1(nn.Module):


    def __init__(self):
        super(ChessNetV1,self).__init__()

        self.convolutionLayer1 = nn.Conv2d(1,6,(8,1))
        self.convolutionLayer2 = nn.Conv2d(1,6,(1,8))   


    #Returns an 8x8x8 vector 
    #           ^^^ ^
    #            |  |
    #   board dims  6pieces+pieceColor+whosTurn       
    def vectorizer(board):
        pieces = {"p":0,"r":1,"b":2,"n":3,"q":4,"k":5,"P":0,"R":1,"B":2,"N":3,"Q":4,"K":5}

        fen = board.fen()
        turn = -1 + 2*int(board.turn==chess.WHITE) 
        i = 0
        c_i = 0
        board_vect = [[] for _ in range(8)]
        while i < 64:
            char = fen[c_i]
            square = [0,0,0,0,0,0,0,turn]

            if char in ["1","2","3","4","5","6","7","8"]:
                for _ in range(int(char)):
                    square = [0,0,0,0,0,0,0,turn]
                    board_vect[int(i/8)].append(square)
                    i += 1
            elif char == " ":
                break
            elif char == "/":
                pass
            else:
                square[pieces[char]] = 1
                square[-2] = -1 + 2*int(char in ascii_uppercase)
                board_vect[int(i/8)].append(square)
                i += 1

            c_i += 1

        return board_vect


if __name__ == "__main__":
    c = ChessNetV1()
    g = chess.Board()

    print(ChessNetV1.vectorizer(g))
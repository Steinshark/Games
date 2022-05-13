import chess
import random
import time
import math
import sys
import numpy
import random
import tensorflow
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import sys 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import json 
from pprint import pp 

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
        return [self.board.uci(move)[-5:] for move in iter(self.board.legal_moves)]

    def push_move(self,move):
        fr = move[:2]
        to = move[3:]

        moves = get_legal_moves()
        if move in moves:
            self.board.push_san(move)
        #Takes in the move 
        elif f"{fr}x{to}":                  
            self.board.push_san(f"{fr}x{to}")
        #Check in the move 
        elif f"{move}+" in moves:
            self.board.push_san(f"{fr}x{to}")
        #Mate in the move 
        elif f"{move}#" in moves:
            self.board.push_san(f"{move}#")
        #Takes with check 
        elif f"{fr}x{to}+" in moves:
            self.board.push_san(f"{fr}x{to}+")
        #Takes with mate 
        elif f"{fr}x{to}#" in moves:
            self.board.push_san(f"{fr}x{to}#")

        else:
            input(f"move: {move} not covered in\n{moves}")

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

        return board_vect

    def get_state_vector_static(board):
        pieces = {"p":0,"r":1,"b":2,"n":3,"q":4,"k":5,"P":6,"R":7,"B":8,"N":9,"Q":10,"K":11}
        fen = board.fen()
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

        if board.turn == chess.WHITE:
            board_vect += [1,0]
        else:
            board_vect += [0,1]

        for color in [chess.WHITE,chess.BLACK]:
            board_vect += [ int(board.has_kingside_castling_rights(color)),
                            int(board.has_queenside_castling_rights(color))]

        return board_vect

    
    def play(self):

            self.board = chess.Board(chess.STARTING_FEN)
            self.game_over = False
            while not self.game_over:
                print(self.board)
                print(f"{self.get_legal_moves()}")
                move = input("mv: ")
                self.board.push_san(move)
                self.check_game()
            res = self.board.outcome()
            print(res)

    def check_move_from_board(board,move):
        return move in [board.uci(move)[-5:] for move in iter(board.legal_moves)]
    
    def check_game(self):
        if self.board.outcome() is None:
            return
        else:
            self.game_over = True

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
        self.output_key = [f"{fr}{to}" for fr in self.squares for to in self.squares if not fr == to]
        self.output_key += [f"{pawn}{sq}{pi}" for pawn in ["a7","b7","c7","d7","d7","e7","f7","g7","h7"] for sq in ["a8","b8","c8","d8","d8","e8","f8","g8","h8"] for pi in ["q","r","b","n"]]
        self.output_key += [f"{pawn}{sq}{pi}" for pawn in ["a2","b2","c2","d2","d2","e2","f2","g2","h2"] for sq in ["a1","b1","c1","d1","d1","e1","f1","g1","h1"] for pi in ["q","r","b","n"]]
        self.build_model()

    def build_model(self):
        self.learning_model = tensorflow.keras.models.Sequential([
            tensorflow.keras.Input(shape=(len(self.input_key),)),
            Dense(1024,activation="relu"),
            Dense(512,activation="relu"),
            Dense(512,activation="relu"),
            Dense(len(self.output_key))])

        self.target_model = tensorflow.keras.models.Sequential([
            tensorflow.keras.Input(shape=(len(self.input_key),)),
            Dense(1024,activation="relu"),
            Dense(512,activation="relu"),
            Dense(512,activation="relu"),
            Dense(len(self.output_key))])

        self.learning_model.compile(loss="huber",optimizer="adam")
        self.target_model.compile(loss="huber",optimizer="adam")

    def train_model(self,iterations,exp_replay=1,discount_factor=.7,simul=10):
        self.exp_replay_step = exp_replay
        self.discount_factor = discount_factor
        simul_games = simul
        # used for experience replay 
        self.experiences =[]
        for i in range(iterations):

            for j in range(simul_games):                
                # Loss is mse of v and z from this game
                t1 = time.time()
                self.experiences += self.monte_carlo_plural([ChessGame() for k in range(j)])
                print(f"FINISHED MCTS size {j} - {time.time()-t1} ")
                              
    def monte_carlo_search(self,game):
        
        odds = {"sample" : .035,#Sample every 30 moves or so
                "experiment": .05} #Play "best move" 95% of time 

        #In our monteCarlo, only expected outcomes are predicted with 
        # the neural net. Thus, no π vector is produced. 
        this_game_experiences = []
        played_move = None


        while game.board.outcome() is None:
            #Get this state and move picker 
            state_vector = game.get_state_vector()
            played_by = game.board.turn
            
            #Either sample to end by playing its best from here 
            if random.uniform(0,1) < odds["sample"]:
                t1 = time.time()
                v,z = self.evaluate_moves_from_here(game,state_vector)
                print(f"ran in {time.time()-t1}")
                this_game_experiences.append((state_vector,v,z))

            #Or play move from here without analysis 
            else:
                if random.uniform(0,1) < odds["experiment"]:
                    played_move = game.random_move()
                    game.board.push_san(played_move)
                else:
                    ordered_moves = numpy.argsort(self.target_model.predict([state_vector])[0])

                    for top_move_index in ordered_moves:
                        if ChessGame.check_move_from_board(game.board,self.output_key[top_move_index]):
                            played_move = self.output_key[top_move_index]
                            game.board.push_san(played_move)
                            break
        result = game.board.outcome()
        if not result.winner == None:  
            if result.winner == played_by:
                val = 1
            else:
                val = -1 
        else:
            val = 0       
        v_vector = list(self.target_model.predict([state_vector]))[0]
        z_vector = [0 for _ in v_vector]
        z_vector[self.output_key.index(played_move)] = val

        this_game_experiences.append([state_vector,v_vector,z_vector])

        return this_game_experiences

    def evaluate_moves_from_here(self,game,state_vector):
        v_vector = list(self.target_model.predict([state_vector])[0])
        z_vector = [0 for _ in self.output_key]
        times = 0
        z = 0 

        trying_new = False
        for i,move in enumerate(v_vector):

            #Try playing every move in position, if not legal, it stays as 0
            if not self.output_key[i] in game.get_legal_moves():
                continue
            z += 1
            exploration_node = game.board.copy()
            was_turn_of = game.board.turn
            exploration_node.push_san(self.output_key[i])
            t1 = time.time()
            z_vector[i] = self.search_till_end(exploration_node,was_turn_of,i) 
            times += (time.time()-t1)
        return v_vector,z_vector

    #None recursive
    def search_till_end(self,game_state,playing_as,gamenum):
        moves = 0
        while game_state.outcome() == None:
            possible_moves = [game_state.uci(move)[-5:] for move in iter(game_state.legal_moves)]
            move_indices = [self.output_key.index(m) for m in possible_moves]
            
            state_vect = ChessGame.get_state_vector_static(game_state)
            vals = self.target_model([tensorflow.constant(state_vect,shape=[1,774])],training=False)[0]
            move_values = {t: vals[t] for t in move_indices}
            max_val = max(move_values,key=move_values.get)
            game_state.push_san(self.output_key[max_val])
            moves += 1
        result = game_state.outcome()
        if not result is None:
            val = 0 
            if not result.winner == None:  
                if result.winner == playing_as: 
                    val = 1
                    print("somebody won!") 
                else: 
                    print("somebody won!") 
                    val = -1 
            else:val = 0 
        return val * self.discount_factor**moves
    
    #Recursive
    def search_till_end2(self,game_state,playing_as,gamenum):
        
        # Either return the actual outcome score
        result = game_state.outcome()
        if not result is None:
            val = 0 
            if not result.winner == None:  
                if result.winner == playing_as: 
                    val = 1
                    print("somebody won!") 
                else: 
                    print("somebody won!") 
                    val = -1 
            else:
                val = 0 
            return val

        # Or keep playing top move (as legal)
        possible_moves = [game_state.uci(move)[-5:] for move in iter(game_state.legal_moves)]
        move_indices = [self.output_key.index(m) for m in possible_moves]
        
        state_vect = ChessGame.get_state_vector_static(game_state)
        vals = self.target_model.predict([state_vect],batch_size=1)[0]
        move_values = {t: vals[t] for t in move_indices}
        max_val = max(move_values,key=move_values.get)
        game_state.push_san(self.output_key[max_val])
        return self.discount_factor * self.search_till_end2(game_state,playing_as,gamenum)

    def monte_carlo_plural(self,games):
        
        odds = {"sample" : .03,#Sample every 30 moves or so
                "experiment": .15} #Play "best move" 95% of time 

        this_game_experiences = []
        played_move = None
        
        games_playing = [g for g in games]
        games_closed = [0 for g in games]
        moves = 0
        while games_playing:
            state_vectors = [g.get_state_vector() for g in games_playing]

            mark_remove = []
            if random.uniform(0,1) < odds['sample']:
                t1 = time.time()
                score_tups = self.evaluate_moves_from_here_plual(games_playing,state_vectors)
                for i,t in enumerate(score_tups):
                    v,z = t
                    this_game_experiences.append([state_vectors[i],v,z])
            
            #Or play move from here without analysis 
            else:
                moves += 1
                if random.uniform(0,1) < odds["experiment"]:
                    for i,game in enumerate(games_playing):
                        played_move = game.random_move()
                        game.board.push_san(played_move)

                        result = game.board.outcome()
                        if not result is None:
                            if not result.winner == None: 
                                game.board.pop() 
                                v,z = self.evaluate_moves_from_here(game,state_vectors[i])

                            this_game_experiences.append([state_vectors[i],v,z])
                            mark_remove.append(i)
                    for i in mark_remove:
                        games_playing.pop(i)
                else:
                    games_moves = [[g.board.uci(move)[-5:] for move in iter(g.board.legal_moves)] for g in games_playing]
                    move_indices = [[self.output_key.index(m) for m in g] for g in games_moves]
                    
                    #Try using this, and predict also
                    vals = self.target_model.predict(state_vectors)

                    move_predictions = [[vals[i][index] for index in move_indices[i]] for i in range(len(move_indices))]
                    top_indices = [move_indices[i][mv.index(max(mv))] for i,mv in enumerate(move_predictions)]
                    top_moves = [self.output_key[ti] for ti in top_indices]

                    for i,game in enumerate(games_playing):
                        played_by = game.board.turn
                        game.board.push_san(top_moves[i])    

                        result = game.board.outcome()
                        if not result is None:
                            print("removing")
                            game.board.pop() 
                            v,z = self.evaluate_moves_from_here(game,state_vectors[i])

                            this_game_experiences.append([state_vectors[i],v,z])
                            mark_remove.append(i)
                    for i in mark_remove:
                        games_playing.pop(i)
        return this_game_experiences
    
    def evaluate_moves_from_here_plual(self,games,state_vectors):
        v_vectors = self.target_model.predict(state_vectors)
        z_vectors = [[0 for i in self.output_key] for g in games]
        
        play_from_positions = [dict() for g in games]

        for i,game in enumerate(games):
            for move in [game.board.uci(move)[-5:] for move in iter(game.board.legal_moves)]:
                exploration_node = game.board.copy()
                was_turn_of = game.board.turn
                exploration_node.push_san(move)
                if not exploration_node.outcome() is None:
                    if not exploration_node.outcome().winner is None:
                        if exploration_node.outcome().winner == was_turn_of:
                            z_vectors[i][self.output_key.index(move)] = 1 
                        else:
                            z_vectors[i][self.output_key.index(move)] = -1
                else:
                    play_from_positions[i][self.output_key.index(move)] = {"state":exploration_node,"turn" :was_turn_of,"score":None}
        
        self.play_multiple_to_end(play_from_positions, z_vectors)
        return zip(v_vectors, [numpy.array(z) for z in z_vectors])
    
    def play_multiple_to_end(self,play_positions,z_vectors):
        #        game#,move#       -> {state,turn,score}
        games = {}
        for game in range(len(play_positions)):
            for spawn_move in play_positions[game]:
                games[game,spawn_move] = play_positions[game][spawn_move]

        while None in [g['score'] for g in games.values()]:    
            games_moves = [[g['state'].uci(move)[-5:] for move in iter(g['state'].legal_moves)] for g in games.values()]
            move_indices = [[self.output_key.index(m) for m in g] for g in games_moves]
            state_vectors = [ChessGame.get_state_vector_static(g['state']) for g in games.values()]
            #Try using this, and predict also
            vals = self.target_model.predict(state_vectors)


            move_predictions = [[vals[i][index] for index in move_indices[i]] for i in range(len(move_indices))]
            top_indices = [move_indices[i][mv.index(max(mv))] for i,mv in enumerate(move_predictions)]
            top_moves = [self.output_key[ti] for ti in top_indices]
            
            moves = 0 
            mark_del = []
            for i,g in enumerate(games):
                game_state = games[g]['state']
                game_state.push_san(top_moves[i])
                result = game_state.outcome() 

                if not result is None:
                    if not result.winner is None:
                        games[g]['score'] = self.discount_factor**moves + (-1 + 2*int(games[g]['turn'] == result.winner))
                    else:
                        games[g]['score'] = 0
                    z_vectors[g[0]][g[1]] = games[g]['score']
                    mark_del.append(g)
            for m in mark_del:
                del(games[m])

            moves += 1

        
if __name__ == "__main__":
    q = QLearning()
    q.train_model(2)
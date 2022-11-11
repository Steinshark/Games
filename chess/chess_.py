#Chess related
import chess
import chess.svg 

#Utility related 
import random
import time
import math
import json
from cairosvg import svg2png

#System related 
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import threading

#Debug related 
from matplotlib import pyplot as plt
from pprint import pp
import tkinter as tk 
from tkinter.scrolledtext import ScrolledText

#Computation related
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


#Class responsible for playing the chess game and interfacing with TF 
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
        #return all moves as UCI (a4b6)
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
        #is move legal?
        return move in [board.uci(move)[-5:] for move in iter(board.legal_moves)]

    def check_game(self):
        if self.board.outcome() is None:
            return
        else:
            self.game_over = True

#Class responsible for doing the learning and training and data collection
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

        #Our input vector is [boolean for piece in square] for each square
        # size 768
        self.input_key = [f"{piece}_on_{square}" for piece in self.pieces for square in self.squares] + ["Wmove","Bmove"]
        self.input_key += ["Wcastlesk","Wcastlesq","Bcastlesk","Bcastlesq"]

        self.output_key = []
        for color in [chess.BLACK,chess.WHITE]:
            for p in [chess.QUEEN,chess.KING,chess.BISHOP,chess.KNIGHT,chess.ROOK,chess.PAWN]:
                piece = chess.Piece(p,color)
                for square in chess.SquareSet(chess.BB_ALL):
                    board = chess.Board()
                    board.clear()
                    board.turn = piece.color
                    board.set_piece_at(square,piece)
                    for p in (board.legal_moves):
                        if not board.uci(p) in self.output_key:
                            self.output_key.append(board.uci(p))
        #white pawn promotions
        self.output_key += ["a7b8q","a7b8r","a7b8b","a7b8n","b7a8q","b7a8r","b7a8b","b7a8n","b7c8q","b7c8r","b7c8b","b7c8n","c7b8q","c7b8r","c7b8b","c7b8n","c7d8q","c7d8r","c7d8b","c7d8n","d7c8q","d7c8r","d7c8b","d7c8n","d7e8q","d7e8r","d7e8b","d7e8n","e7d8q","e7d8r","e7d8b","e7d8n","e7f8q","e7f8r","e7f8b","e7f8n","f7e8q","f7e8r","f7e8b","f7e8n","f7g8q","f7g8r","f7g8b","f7g8n","g7f8q","g7f8r","g7f8b","g7f8n","g7h8q","g7h8r","g7h8b","g7h8n","h7g8q","h7g8r","h7g8b","h7g8n",]
        #black pawn promotions
        self.output_key += ["a2b1q","a2b1r","a2b1b","a2b1n","b2a1q","b2a1r","b2a1b","b2a1n","b2c1q","b2c1r","b2c1b","b2c1n","c2b1q","c2b1r","c2b1b","c2b1n","c2d1q","c2d1r","c2d1b","c2d1n","d2c1q","d2c1r","d2c1b","d2c1n","d2e1q","d2e1r","d2e1b","d2e1n","e2d1q","e2d1r","e2d1b","e2d1n","e2f1q","e2f1r","e2f1b","e2f1n","f2e1q","f2e1r","f2e1b","f2e1n","f2g1q","f2g1r","f2g1b","f2g1n","g2f1q","g2f1r","g2f1b","g2f1n","g2h1q","g2h1r","g2h1b","g2h1n","h2g1q","h2g1r","h2g1b","h2g1n",]
        self.build_model()

    def build_model(self):

        if os.path.exists("model"):
            self.learning_model = tensorflow.keras.models.load_model("model")
        else:
            self.learning_model = tensorflow.keras.models.Sequential([
                tensorflow.keras.layers.InputLayer(input_shape=(len(self.input_key),)),
                Dense(2048,activation='relu'),
                Dense(512,activation="relu"),
                Dense(512,activation="relu"),
                Dense(len(self.output_key))])

        self.learning_model.compile(
                loss=tk.keras.losses.MeanSquaredError(),
                optimizer=tensorflow.keras.optimizers.Adam(learning_rate=.003))

    def train_model(self,iterations,exp_replay=1,discount_factor=.7,simul=10,output=None):
        self.exp_replay_step = exp_replay
        self.discount_factor = discount_factor
        simul_games = simul
        # used for experience replay
        times = []
        st = time.time()

        t0 = time.time()
        for iters in range(iterations):
            
            self.experiences = []
            if not output is None:
                output.insert(tk.END,f"Begin training seq {iters+1}/{iterations}\n")
            else:
                print(f"Begin training seq {iters+1}/{iterations}")
            for j in range(exp_replay):
                t1 = time.time()
                self.experiences += self.monte_carlo_plural([ChessGame() for k in range(simul)])
                if not output is None:
                    output.insert(tk.END,f"\tCollection {j+1}/{exp_replay}\t{simul} games in {(time.time()-t1):.2f}\n")
                else:
                    print(f"\tCollection {j+1}/{exp_replay}\t{simul} games in {(time.time()-t1):.2f}")
           
            if not output is None:
                output.insert(tk.END,f"\ttraining on dataset size {len(self.experiences)}\n")
            else:
                print(f"\ttraining on dataset size {len(self.experiences)}")

            #Save the experiences
            writing = []
            for exp in self.experiences:
                a,z,v = exp

                writing.append((a,z,v))
            numpy.save(f"experiences{len(self.experiences)}",numpy.array(writing,dtype=object))

            #Then train on the batch

            x_train = []
            y_train = []
            for tup in writing:
                x_item = numpy.array(tup[0])
                y_item = tup[1]
                x_train.append(x_item)
                y_train.append(y_item)

            x_train = numpy.array(x_train)
            y_train = numpy.array(y_train)
            self.learning_model.fit(x_train,y_train,batch_size=8)
            if not output is None:
                output.insert(tk.END,f"\ttrained model on experience set\n")
            else:
                print(f"\ttrained model on experience set")
        self.learning_model.save("model")

    def evaluate_moves_from_here(self,game,state_vector):
        v_vector = list(self.learning_model(tensorflow.constant([state_vector])))
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
        return v_vector,numpy.array(z_vector)

    def search_till_end(self,game_state,playing_as,gamenum):
        moves = 0
        while game_state.outcome() == None:
            possible_moves = [game_state.uci(move)[-5:] for move in iter(game_state.legal_moves)]
            move_indices = [self.output_key.index(m) for m in possible_moves]

            state_vect = ChessGame.get_state_vector_static(game_state)
            vals = self.learning_model([tensorflow.constant(state_vect,shape=[1,774])],training=False)[0]
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

    def monte_carlo_plural(self,games):

        odds = {"sample" : .01,#Sample every 30 moves or so
                "experiment": .15} #Play "best move" 95% of time

        this_game_experiences = []
        played_move = None

        games_playing = [g for g in games]
        games_closed = [0 for g in games]
        moves = 0


        self.global_predictions = 0
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
                if random.uniform(0,1) < odds["experiment"]:
                    for i,game in enumerate(games_playing):
                        played_move = game.random_move()
                        game.board.push_san(played_move)

                        result = game.board.outcome()
                        if not result is None:
                            if not result.winner == None:
                                game.board.pop()
                            score_tups = self.evaluate_moves_from_here_plual([game],[state_vectors[i]])
                            for e,t in enumerate(score_tups):
                                v,z = t
                                this_game_experiences.append([state_vectors[e],v,z])

                            mark_remove.append(games_playing[i])
                    for i in mark_remove:
                        games_playing.remove(i)
                else:
                    games_moves = [[g.board.uci(move)[-5:] for move in iter(g.board.legal_moves)] for g in games_playing]
                    move_indices = [[self.output_key.index(m) for m in g] for g in games_moves]

                    #Try using this, and predict also
                    vals = self.learning_model(tensorflow.constant(state_vectors),training=False)

                    move_predictions = [[vals[i][index] for index in move_indices[i]] for i in range(len(move_indices))]
                    top_indices = [move_indices[i][mv.index(max(mv))] for i,mv in enumerate(move_predictions)]
                    top_moves = [self.output_key[ti] for ti in top_indices]

                    for i,game in enumerate(games_playing):
                        played_by = game.board.turn
                        game.board.push_san(top_moves[i])

                        result = game.board.outcome()
                        if not result is None:
                            game.board.pop()
                            score_tups = self.evaluate_moves_from_here_plual([game],[state_vectors[i]])
                            for e,t in enumerate(score_tups):
                                v,z = t
                                this_game_experiences.append([state_vectors[e],v,z])
                            mark_remove.append(games_playing[i])
                    for i in mark_remove:
                        games_playing.remove(i)
            moves += 1

        return this_game_experiences

    def evaluate_moves_from_here_plual(self,games,state_vectors):
        v_vectors = self.learning_model.predict(tensorflow.constant(state_vectors),batch_size=len(state_vectors))
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
        moves = 0
        t1 = time.time()
        while None in [g['score'] for g in games.values()]:
            games_moves = [[g['state'].uci(move)[-5:] for move in iter(g['state'].legal_moves)] for g in games.values()]
            move_indices = [[self.output_key.index(m) for m in g] for g in games_moves]
            state_vectors = tensorflow.constant([ChessGame.get_state_vector_static(g['state']) for g in games.values()])
            #Try using this, and predict also
            vals = self.learning_model.predict(state_vectors,batch_size=len(games))


            move_predictions = [[vals[i][index] for index in move_indices[i]] for i in range(len(move_indices))]
            top_indices = [move_indices[i][mv.index(max(mv))] for i,mv in enumerate(move_predictions)]
            top_moves = [self.output_key[ti] for ti in top_indices]

            mark_del = []
            for i,g in enumerate(games):
                game_state = games[g]['state']
                game_state.push_san(top_moves[i])
                result = game_state.outcome()

                if not result is None:
                    if not result.winner is None:
                        games[g]['score'] = (self.discount_factor**(moves/10)) * (-1 + 2*int(games[g]['turn'] == result.winner))
                    else:
                        games[g]['score'] = 0
                    z_vectors[g[0]][g[1]] = games[g]['score']
                    mark_del.append(g)
            for m in mark_del:
                del(games[m])

            moves += 1

    def run_as_ui(self):
        window = tk.Tk()
        mainframe = tk.Frame(window)

        #TrainingBox 
        training_box = tk.Frame(window)

        train_label = tk.Label(training_box,text="Training Dashboard")
        train_label.grid(row=0,column=0,columnspan=2,sticky='ew')

        iter_label = tk.Label(training_box,text="iters:") 
        exp_label = tk.Label(training_box,text="experience:") 
        simul_label = tk.Label(training_box,text="simul:") 
        
        iter_entry = tk.Entry(training_box)
        exp_entry = tk.Entry(training_box)
        simul_entry = tk.Entry(training_box)

        iter_label. grid(row=1,column=0,sticky="ew")
        exp_label.  grid(row=2,column=0,sticky="ew")
        simul_label.grid(row=3,column=0,sticky="ew")

        iter_entry. grid(row=1,column=1,stick="ew")
        exp_entry.  grid(row=2,column=1,sticky="ew")
        simul_entry.grid(row=3,column=1,sticky="ew")
        out_box = tk.Frame(window)
        output_view = ScrolledText(out_box)

        train_button = tk.Button(training_box,text='Train!',command=lambda:self.run_model(int(iter_entry.get()),int(exp_entry.get()),int(simul_entry.get()),output=output_view))
        train_button.grid(row=4,column=0,columnspan=2,sticky="ew")
        
        training_box.grid(row=0,column=0)

        #train output
        out_label = tk.Label(out_box,text="Program Out")
        out_label.grid(row=0,column=0,stick="ew")
        output_view.grid(row=1,column=0,stick="ew")

        out_box.grid(row=1,column=0)

        #Playing Box 
        play_box = tk.Frame(window)

        play_label = tk.Label(play_box,text="Game Dashboard")
        self.move_entry = tk.Entry(play_box)
        self.start_new = tk.Button(play_box,text='New Game',command=lambda: self.reset_game())
        self.end_res = tk.Label(play_box,text="Game result")
        game_play   = tk.Button(play_box,text='Play',command = lambda: self.play_move())

        play_label.grid(row=0,column=0,columnspan=2)
        game_play.grid(row=1,column=0,sticky="ew")
        self.move_entry.grid(row=1,column=1,sticky="ew")
        self.start_new.grid(row=2,column=0,sticky="ew")
        self.end_res.grid(row=2,column=1,sticky="ew")
        play_box.grid(row=0,column=1)


        #Game out 
        game_out = tk.Frame(window)

        self.game_canvas = tk.Canvas(game_out,height=500,width=500)
        self.game_canvas.grid(row=0,column=0,sticky="ew")

        game_out.grid(row=1,column=1)


        self.play_board = chess.Board()
        self.game_canvas.create_image(20,20,image=self.chess_png(self.play_board),anchor="nw")
        #Finish up and run

        mainframe.columnconfigure(0,weight=1)
        mainframe.columnconfigure(1,weight=1)

        mainframe.rowconfigure(0,weight=1)
        mainframe.rowconfigure(1,weight=1)
        mainframe.grid(row=0,column=0)
        window.mainloop()

    def chess_png(self,board):
        svg_raw =  chess.svg.board(board)
        png_file = svg2png(bytestring=svg_raw,write_to="current_board.png")
        self.img = tk.PhotoImage(file="current_board.png")
        return self.img

    def play_move(self):

        #My move
        try:
            self.play_board.push_uci(self.move_entry.get())
        except ValueError:
            self.move_entry.text = 'Bad move!'
            return

        if not self.play_board.outcome() is None:
            winner = self.play_board.outcome().winner
            if winner == chess.WHITE:
                self.end_res["text"] = "White wins"
            elif winner == chess.BLACK:
                self.end_res["text"] = "Black wins"
            else:
                self.end_res['text'] = f"{self.play_board.outcome().termination}"
            self.game_canvas.create_image(20,20,image=self.chess_png(self.play_board),anchor="nw")
            return


        #Engine move 
        moves = [self.play_board.uci(move)[-5:] for move in iter(self.play_board.legal_moves)]
        move_indices = [self.output_key.index(m) for m in moves]

        #Try using this, and predict also
        vals = self.learning_model(tensorflow.constant([ChessGame.get_state_vector_static(self.play_board)]),training=False)[0]

        move_predictions = [vals[i] for i in move_indices]
        top_index = move_indices[move_predictions.index(max(move_predictions))]
        top_move = self.output_key[top_index]
        self.play_board.push_uci(top_move)

        if not self.play_board.outcome() is None:
            winner = self.play_board.outcome().winner
            if winner == chess.WHITE:
                self.end_res["text"] = "White wins"
            elif winner == chess.BLACK:
                self.end_res["text"] = "Black wins"
            else:
                self.end_res['text'] = f"{self.play_board.outcome().termination}"
            self.game_canvas.create_image(20,20,image=self.chess_png(self.play_board),anchor="nw")
            return

        self.game_canvas.create_image(20,20,image=self.chess_png(self.play_board),anchor="nw")

    def reset_game(self):
        self.play_board = chess.Board()
        self.game_canvas.create_image(20,20,image=self.chess_png(self.play_board),anchor="nw")

    def run_model(self,i,e,s,output=None):
        t = threading.Thread(target=self.train_model,args=[i],kwargs={"exp_replay":e,"simul":s,"output":output})
        t.start()


if __name__ == "__main__":  
    q = QLearning()
    #q.run_as_ui()
    g = [1,5,10,15,20,25,30,50,75,100,125,150,175,200]
    times = [[],[],[]] 

    for g_i in g:
        for i in [0,1,2]:
            t1 = time.time()
            q.train_model(iterations=1,exp_replay=1,simul=g_i)
            times[i].append(time.time()-t1 )
    plt.plot(g,times[0])
    plt.plot(g,times[1])
    plt.plot(g,times[2])
    plt.show()
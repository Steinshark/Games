#Chess related
import chess
import chess.svg 

#Utility related 
import random
import time
import math
import json
from cairosvg import svg2png
from sklearn.utils import extmath 
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
import torch 
sys.path.append("C:/gitrepos")
from steinpy.ml.networks import ChessNetCompat
from steinpy.ml.rl import Chess, Tree,Node
#Computation related
import numpy
#print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


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

	def build_model(self,gen=0):

		if os.path.exists(f"C:/data/chess/models/gen{gen}"):
			self.model       = ChessNetCompat(n_ch=13)
			self.model.load_state_dict(torch.load(f"C:/data/chess/models/gen{gen}"))
			self.model      = torch.jit.script(self.model,[torch.randn(1,13,8,8)])
			return 
		else:
			print(f"no model found with gen {gen}")
			self.model       = ChessNetCompat(n_ch=13)

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


		self.play_board = chess.Board(fen="5B1k/8/7K/8/pp6/2b4P/2b5/6q1 w - - 0 1")
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
	
	@staticmethod
	def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]

	def play_move(self):
		softmax             = torch.nn.Softmax(dim=0)
		move_indices            = list(range(1968))
		#My move
		try:
			self.play_board.push_uci(self.move_entry.get())
		except ValueError:
			self.move_entry.text = 'Bad move!'
			return

		if not self.play_board.outcome() is None:
			winner = self.play_board.result()
			if winner == chess.WHITE:
				self.end_res["text"] = self.play_board.result()
			elif winner == chess.BLACK:
				self.end_res["text"] = self.play_board.result()
			else:
				self.end_res['text'] = f"{self.play_board.outcome().termination}"
			self.game_canvas.create_image(20,20,image=self.chess_png(self.play_board),anchor="nw")
			return


		#Engine move 
		tree 			= Tree(self.play_board,self.model,None,draw_thresh=250)

		#Try using this, and predict also
		t0 = time.time()
		local_policy 	= tree.get_policy(search_iters=2000,abbrev=False)
		print(f"built policy in {(time.time() - t0):.2f}s\n{[ (Chess.chess_moves[k],v.num_visited) for k,v in tree.root.children.items()]}")
		local_softmax 	= QLearning.softmax(numpy.asarray(list(local_policy.values())))
		for key,prob in zip(local_policy.keys(),local_softmax):
			local_policy[key] = prob
		pi              = numpy.zeros(1968)
		for move_i,prob in local_policy.items():
			pi[move_i]    = prob 
		
		#sample move from policy 
		next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
		next_move               = Chess.index_to_move[next_move_i]
		self.play_board.push(next_move)

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
	q.build_model()
	q.run_as_ui()
	
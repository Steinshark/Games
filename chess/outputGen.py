import chess 



output_key = []

for piece in [chess.Piece(chess.PAWN,chess.BLACK),chess.Piece(chess.PAWN,chess.WHITE),chess.Piece(chess.KNIGHT,chess.BLACK),chess.Piece(chess.KNIGHT,chess.WHITE), chess.Piece(chess.ROOK,chess.BLACK),chess.Piece(chess.ROOK,chess.WHITE),chess.Piece(chess.BISHOP,chess.BLACK),chess.Piece(chess.BISHOP,chess.WHITE),chess.Piece(chess.QUEEN,chess.BLACK),chess.Piece(chess.QUEEN,chess.WHITE),chess.Piece(chess.KING,chess.BLACK),chess.Piece(chess.KING,chess.WHITE)]:
    board = chess.Board()
    for square in chess.SquareSet(chess.BB_ALL):
        board.set_piece_at(square,piece)
        fr = chess.square_name(square)
        for p in board.attacks(square):
            if not f"{fr}{chess.square_name(p)}" in output_key:
                output_key.append(f"{fr}{chess.square_name(p)}")



print(len(output_key))


["a7a8q","a7a8r","a7a8b","a7a8k","a7b8q","a7b8r","a7b8b","a7b8k","b7a8q","b7a8r","b7a8b","b7a8k","b7b8q","b7b8r","b7b8b","b7b8k","b7c8q","b7c8r","b7c8b","b7c8k","c7b8q","c7b8r","c7b8b","c7b8k","c7c8q","c7c8r","c7c8b","c7c8k","c7d8q","c7d8r","c7d8b","c7d8k","d7c8q","d7c8r","d7c8b","d7c8k","d7d8q","d7d8r","d7d8b","d7d8k","d7e8q","d7e8r","d7e8b","d7e8k","e7d8q","e7d8r","e7d8b","e7d8k","e7e8q","e7e8r","e7e8b","e7e8k","e7f8q","e7f8r","e7f8b","e7f8k","f7e8q","f7e8r","f7e8b","f7e8k","f7f8q","f7f8r","f7f8b","f7f8k","f7g8q","f7g8r","f7g8b","f7g8k","g7f8q","g7f8r","g7f8b","g7f8k","g7g8q","g7g8r","g7g8b","g7g8k","g7h8q","g7h8r","g7h8b","g7h8k","h7g8q","h7g8r","h7g8b","h7g8k","h7h8q","h7h8r","h7h8b","h7h8k","a2a1q","a2a1r","a2a1b","a2a1k","a2b1q","a2b1r","a2b1b","a2b1k","b2a1q","b2a1r","b2a1b","b2a1k","b2b1q","b2b1r","b2b1b","b2b1k","b2c1q","b2c1r","b2c1b","b2c1k","c2b1q","c2b1r","c2b1b","c2b1k","c2c1q","c2c1r","c2c1b","c2c1k","c2d1q","c2d1r","c2d1b","c2d1k","d2c1q","d2c1r","d2c1b","d2c1k","d2d1q","d2d1r","d2d1b","d2d1k","d2e1q","d2e1r","d2e1b","d2e1k","e2d1q","e2d1r","e2d1b","e2d1k","e2e1q","e2e1r","e2e1b","e2e1k","e2f1q","e2f1r","e2f1b","e2f1k","f2e1q","f2e1r","f2e1b","f2e1k","f2f1q","f2f1r","f2f1b","f2f1k","f2g1q","f2g1r","f2g1b","f2g1k","g2f1q","g2f1r","g2f1b","g2f1k","g2g1q","g2g1r","g2g1b","g2g1k","g2h1q","g2h1r","g2h1b","g2h1k","h2g1q","h2g1r","h2g1b","h2g1k","h2h1q","h2h1r","h2h1b","h2h1k",]



a2a3
b2b3
c2c3
d2d3
e2e3
f2f3
g2g3
h2h3
a2a4
b2b4
c2c4
d2d4
e2e4
f2f4
g2g4
h2h4
a3a4
b3b4
c3c4
d3d4
e3e4
f3f4
g3g4
h3h4
a4a5
b4b5
c4c5
d4d5
e4e5
f4f5
g4g5
h4h5
a5a6
b5b6
c5c6
d5d6
e5e6
f5f6
g5g6
h5h6
a6a7
b6b7
c6c7
d6d7
e6e7
f6f7
g6g7
h6h7

a3a2
b3b2
c3c2
d3d2
e3e2
f3f2
g3g2
h3h2
a4a3
b4b3
c4c3
d4d3
e4e3
f4f3
g4g3
h4h3
a5a4
b5b4
c5c4
d5d4
e5e4
f5f4
g5g4
h5h4
a6a5
b6b5
c6c5
d6d5
e6e5
f6f5
g6g5
h6h5
a7a5
b7b5
c7c5
d7d5
e7e5
f7f5
g7g5
h7h5
a7a6
b7b6
c7c6
d7d6
e7e6
f7f6
g7g6
h7h6
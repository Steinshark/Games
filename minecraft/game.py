import pyglet 
import random 
import math 


class World:
    def __init__(self):
        pass


class Chunk:
    def __init__(self):
        pass


class Block:
    def __init__(self, coordinate, blockType):
        self.type = blockType 
        self.fll = coordinate
        self.coords = buildCoordinatesFromFLL()
        self.TopSurface = list(self.coords[location] for location in ['ful','fur','rul','rur'])


    def buildCoordinatesFromFLL():
        CoordinateDictionary =  {
                                    'fll' : self.fll,
                                    'flr' : self.fll + Coord(1,0,0),
                                    'ful' : self.fll + Coord(0,1,0),
                                    'fur' : self.fll + Coord(1,1,0),
                                    'rll' : self.fll + Coord(0,0,-1),
                                    'rlr' : self.fll + Coord(1,0,-1),
                                    'rul' : self.fll + Coord(0,1,-1),
                                    'rur' : self.fll + Coord(1,1,-1)
                                }

class Coordinate:
    def __init__(self,x,y,z):
        self.x = x 
        self.y = y
        self.z = z

    def __add__(self,c2):
        return Coord(self.x + c2.x, self.y + c2.y, + self.z + c2.z)

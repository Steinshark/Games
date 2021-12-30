import pyglet
import random
import math
from makeTopoMap import *

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
        self.coords = Block.buildCoordinatesFromFLL(self.fll)
        self.TopSurface = list((self.coords[location].x,
                                self.coords[location].y,
                                self.coords[location].z,
                                ) for location in ['ful','fur','rul','rur'])
        self.AllPoints = [ (self.coords[location].x,self.coords[location].y,self.coords[location].z)
                                for location in self.coords]

        self.Wireframe = self.buildWireframe()

    def buildWireframe(self):
        wireframe_pairs = []
        listp1 = [[self.coords[p1].x,self.coords[p1].y,self.coords[p1].z] for p1 in ['ful','ful','fur','rur',    'fll','fll','flr','rlr',    'ful','fur','rur','rul' ]]
        listp2 = [[self.coords[p2].x,self.coords[p2].y,self.coords[p2].z] for p2 in ['fur','rul','rur','rul',    'flr','rll','rlr','rll',    'fll','rur','rlr','rll' ]]

        for p1,p2 in zip(listp1,listp2):
            wireframe_pairs.append(tuple(p1))
            wireframe_pairs.append(tuple(p2))
        return wireframe_pairs
    def buildCoordinatesFromFLL(fll):
        CoordinateDictionary =  {
                                    'fll' : fll,
                                    'flr' : fll + Coordinate(1,0,0),
                                    'ful' : fll + Coordinate(0,1,0),
                                    'fur' : fll + Coordinate(1,1,0),
                                    'rll' : fll + Coordinate(0,0,-1),
                                    'rlr' : fll + Coordinate(1,0,-1),
                                    'rul' : fll + Coordinate(0,1,-1),
                                    'rur' : fll + Coordinate(1,1,-1)
                                }
        return CoordinateDictionary

class Coordinate:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self,c2):
        return Coordinate(self.x + c2.x, self.y + c2.y, + self.z + c2.z)

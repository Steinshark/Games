from world import *
from random import randint, uniform
from pyglet import window, gl
import pyglet
from math import cos, sin, sqrt, pi
from pyglet.window import mouse, key
from pyglet.gl import   GL_PROJECTION, glClear, GL_MODELVIEW, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_COLOR_BUFFER_BIT,\
                        glLoadIdentity, glViewport, glEnableClientState, GL_VERTEX_ARRAY, glMatrixMode, gluPerspective, glEnable, glBlendFunc,\
                        glFrustum, GL_DEPTH_BUFFER_BIT, gluLookAt
from pyglet.window.key import *
from primitives import draw_line_loop, draw_line, draw_points


class Game:
    def __init__(self):
        # Create a collection of settings for future reference throughout the game
        self.settings = {
            "dimension"     : 10,
            "width"         : 800,
            "height"        : 640,
            "resize"        : True,
            "caption"       : "testing",
            "dimension"     : 10
            }
        self.camera = { "eye"       :   {'x': 0.0, 'y' : 0.0, 'z' : 0.0},
                        "center"    :   {'x': 0.0, 'y' : 0.0, 'z' : 10.0},
                        "up"        :   {'x': 0.0, 'y' : 1.0, 'z' : 0.0},
                        "near"      :   1,
                        "far"       :   100
                        }
        self.camera_vector = {  "angle_inclination" : 0,
                                "angle_horizontal"  : 0,
                                "length"            : 10

        }

        # INIT the pyglet window for the game
        self.window = pyglet.window.Window(width=self.settings["width"],height=self.settings["height"],resizable=self.settings["resize"],caption=self.settings["caption"])
        # Fix transparency things
        #glEnable(GL_BLEND)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0,self.settings['width']/self.settings['height'],self.camera['near'],self.camera['far']);
        glMatrixMode(GL_MODELVIEW)



        #gluLookAt(  self.camera['eye']['x'],self.camera['eye']['y'],self.camera['eye']['z'],
        #self.camera['center']['x'],self.camera['center']['y'],self.camera['center']['z'],
        #self.camera['up']['x'],self.camera['up']['y'],self.camera['up']['z'])
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glViewport(0, 0, self.settings["width"], self.settings["height"])



        # Create a 3D list of blocks for testing purposes
        # Index blocks as blocks[x][y][z]
        self.blocks_dictionary = {x : {y : {z : Block(Coordinate(x,y,z),"grass") for z in range(self.settings["dimension"])} for y in range(self.settings["dimension"])} for x in range(self.settings["dimension"])}



        # AKA THIS IS THE GAME LOOP
        @self.window.event
        def on_draw():
            self.window.clear()
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(60.0,self.settings['width']/self.settings['height'],self.camera['near'],self.camera['far']);
            gluLookAt(  self.camera['eye']['x'],self.camera['eye']['y'],self.camera['eye']['z'],
                        self.camera['center']['x'],self.camera['center']['y'],self.camera['center']['z'],
                        self.camera['up']['x'],self.camera['up']['y'],self.camera['up']['z'])
            for x in self.blocks_dictionary:
                for y in self.blocks_dictionary[x]:
                    for z in self.blocks_dictionary[x][y]:
                        block = self.blocks_dictionary[x][y][z]
                        block_points = block.allPoints
                        draw_points(block_points)
            print(f"CAMERA: x:{self.camera['eye']['x']} y: {self.camera['eye']['y']} z: {self.camera['eye']['z']}\nCENTER: x:{self.camera['center']['x']} y: {self.camera['center']['y']} z: {self.camera['center']['z']}\n")


        @self.window.event
        def on_key_press(symbol,modifyer):
            if symbol == W:
                self.camera['eye']['z'] += 1
                self.camera['center']['z'] += 1
            elif symbol == S:
                self.camera['eye']['z'] -= 1
                self.camera['center']['z'] -= 1
            elif symbol == A:
                self.camera['eye']['x'] += 1
                self.camera['center']['x'] += 1
            elif symbol == D:
                self.camera['eye']['x'] -= 1
                self.camera['center']['x'] -= 1

        @self.window.event
        def on_mouse_press(x, y, button, modifiers):
            pass

        @self.window.event
        def on_mouse_drag(x, y, dx, dy, button, modifiers):
            pass
        @self.window.event
        def on_mouse_motion(x, y, dx, dy):
            self.camera_vector["angle_horizontal"]  += dx * .01
            self.camera_vector["angle_inclination"] += dy * .01

            self.camera_vector["angle_horizontal"]  = self.camera_vector["angle_horizontal"]        % (2 * pi)
            self.camera_vector["angle_inclination"] = self.camera_vector["angle_inclination"]       % (2 * pi)

            self.compute_camera_angle()



    def run_game(self):
        pyglet.app.run()

    # Calculate the position of the "center", or where the camera should
    # Be pointing to
    def compute_camera_angle(self):

        theta   = self.camera_vector["angle_horizontal"]
        phi     = self.camera_vector["angle_inclination"]


        print(f"theta: {theta} phi: {phi}")
        self.camera["center"]["x"] = self.camera['eye']['x'] + 10.0 * sin(phi) * cos(theta)
        self.camera["center"]["z"] = self.camera['eye']['z'] + 10.0 * sin(phi) * sin(theta)


        self.camera['center']['y'] = self.camera['eye']['y'] + 10.0 * cos(phi)

if __name__ == "__main__":
    game = Game()
    game.run_game()

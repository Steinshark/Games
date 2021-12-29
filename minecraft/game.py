#local dependencies
from world import *
from primitives import draw_line_loop, draw_lines, draw_points

#mechanics dependencies
from random import randint, uniform
from time import time
from math import cos, sin, sqrt, pi

#Graphical dependencies
import pyglet
from pyglet.window import mouse, key
from pyglet.clock import tick
from pyglet.gl import   GL_PROJECTION, glClear, GL_MODELVIEW, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_COLOR_BUFFER_BIT,\
                        glLoadIdentity, glViewport, glEnableClientState, GL_VERTEX_ARRAY, glMatrixMode, gluPerspective, glEnable, glBlendFunc,\
                        glFrustum, GL_DEPTH_BUFFER_BIT, gluLookAt
from pyglet.window.key import *



class Game:
    # Scheduled methods must be declared first
    def update(self,dt):
        pass
    def tick(self, dt):
        self.game_settings['frame_time']    = time() - self.game_settings['time']
        self.game_settings['time'] += self.game_settings['frame_time']
        print(f"\n\nFrametime: {self.game_settings['frame_time']}\n\nFramerate: {1.0/self.game_settings['frame_time']}")


    def __init__(self):
        ########################################################################
        ##################### GAME ENVIRONEMNT DECLARATIONS ####################
        ########################################################################
            # Contains a collection of general settings
            self.settings =             {
                "width"             : 800,
                "height"            : 640,
                "resize"            : True,
                "caption"           : "testing",
                }
            # Create a collection of coordinates to point the camera
            self.camera =               {
                "eye"               :   {'x': 0.0, 'y' : 0.0, 'z' : 0.0},
                "center"            :   {'x': 0.0, 'y' : 0.0, 'z' : 10.0},
                "up"                :   {'x': 0.0, 'y' : 1.0, 'z' : 0.0},
                "near"              :   1,
                "far"               :   100
                }
            # Contains camera math components
            self.camera_vector =        {
                "angle_inclination" : 0,
                "angle_horizontal"  : 0,
                "length"            : 10

                }
            # Contains all the game mechanical information
            self.mechanics =            {
                "clock"             : 0,
                "fps"               : 60
            }
            # Contains all methods that will be called mapped to their calling interval
            self.scheduled_functions =  {
                self.update         : 1 / self.mechanics['fps'],
                self.tick           : 1 / self.mechanics['fps']
            }
            # Contains the gameplay component variables
            self.game_settings =      {
                "dimension"         : 10,
                "time"              : 0.0,
                "start_time"        : 0.0,
                "frame_time"        : 0.0,
                "keyboard"          : {}
            }
            # Contains the gameplay items that will be instantiated
            self.game_components =      {
                "blocks"         : None
            }


        ########################################################################
        ######################### GRAPHICAL SETUP WORK ### #####################
        ########################################################################
            # init the pyglet window for the game
            self.window = pyglet.window.Window(width=self.settings["width"],height=self.settings["height"],resizable=self.settings["resize"],caption=self.settings["caption"])
            # SETUP the 3d Environment
            glEnable(GL_BLEND)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(60.0,self.settings['width']/self.settings['height'],self.camera['near'],self.camera['far']);
            glMatrixMode(GL_MODELVIEW)


        ########################################################################
        ###################### GAME ENVIRONEMNT CREATION  ######################
        ########################################################################
            # Create a 3D dict of blocks referenced by xyz coordinate
            self.game_components['blocks'] = {x : {y : {z : Block(Coordinate(x,y,z),"grass") for z in range(self.game_settings["dimension"])} for y in range(1)} for x in range(self.game_settings["dimension"])}
            # Schedule all automatic method calls
            for function_call in self.scheduled_functions:
                pyglet.clock.schedule_interval(function_call, self.scheduled_functions[function_call])
            # Start the world clocks
            self.game_settings['start_time'] = time()

            
        ########################################################################
        ###################### DECORATED METHODS CREATION  #####################
        ########################################################################
            @self.window.event
            def on_draw():
                self.window.clear()
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(60.0,self.settings['width']/self.settings['height'],self.camera['near'],self.camera['far']);
                gluLookAt(  self.camera['eye']['x'],self.camera['eye']['y'],self.camera['eye']['z'],
                            self.camera['center']['x'],self.camera['center']['y'],self.camera['center']['z'],
                            self.camera['up']['x'],self.camera['up']['y'],self.camera['up']['z'])

                self.draw_world()
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

    def draw_world(self):
        for x in self.game_components['blocks']:
            for y in self.game_components['blocks'][x]:
                for z in self.game_components['blocks'][x][y]:
                    block = self.game_components['blocks'][x][y][z]
                    block_points = block.wireframe
                    draw_lines(block_points)
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

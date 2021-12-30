#local dependencies
from world              import *
from primitives         import draw_line_loop, draw_lines, draw_points, draw_triangle_fan, GL_POINTS

#mechanics dependencies
from random             import randint, uniform
from time               import time
from math               import cos, sin, sqrt, pi
from ctypes             import pointer, sizeof
from array import array

#Graphical dependencies
import pyglet
from pyglet.window      import mouse, key
from pyglet.clock       import tick
from pyglet.window.key  import *
from pyglet.gl          import   GL_PROJECTION, glClear, GL_MODELVIEW, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_COLOR_BUFFER_BIT,\
                        glLoadIdentity, glViewport, glEnableClientState, GL_VERTEX_ARRAY, glMatrixMode, gluPerspective, glEnable, glBlendFunc,\
                        glFrustum, GL_DEPTH_BUFFER_BIT, gluLookAt, glTranslatef, glRotatef, GLuint, glGenBuffers, glBindBuffer, glBufferData,\
                        GL_ARRAY_BUFFER, GL_STATIC_DRAW, GLfloat, glDrawArrays, glVertexPointer, GL_FLOAT, GL_LINES




class Game:
    # Scheduled methods must be declared first
    def update(self,dt):
        pass
    def tick(self, dt):
        self.game_settings['frame_time']    = time() - self.game_settings['time']
        self.game_settings['time'] += self.game_settings['frame_time']


    def __init__(self):
            pyglet.gl.ERROR_CHECKING = False
        ########################################################################
        ##################### GAME ENVIRONEMNT DECLARATIONS ####################
        ########################################################################
            # Contains a collection of general settings
            self.settings               =   {
                "width"             : 800,
                "height"            : 640,
                "resize"            : True,
                "caption"           : "testing",
                }
            # Create a collection of coordinates to point the camera
            self.camera                 =   {
                "eye"               :   {'x': 0.0, 'y' : 0.0, 'z' : 0.0},
                "center"            :   {'x': 0.0, 'y' : 0.0, 'z' : 10.0},
                "up"                :   {'x': 0.0, 'y' : 1.0, 'z' : 0.0},
                "near"              :   1,
                "far"               :   100
                }
            # Contains camera math components
            self.camera_vector          =   {
                "angle_inclination" : pi / 2 ,
                "angle_horizontal"  : 0,
                "length"            : 10

                }
            # Contains all the game mechanical information
            self.mechanics              =   {
                "clock"             : 0,
                "fps"               : 60,
                "player_step"       : .1
            }
            # Contains all methods that will be called mapped to their calling interval
            self.scheduled_functions    =   {
                self.update         : 1 / self.mechanics['fps'],
                self.tick           : 1 / self.mechanics['fps']
            }
            # Contains the gameplay component variables
            self.game_settings          =   {
                "dimension"         : 100,
                "time"              : 0.0,
                "start_time"        : 0.0,
                "frame_time"        : 0.0,
                "keyboard"          : {}
            }
            # Contains the gameplay items that will be instantiated
            self.game_components        =   {
                "blocks"            :   None
            }
            # Contians al inputs the game is currently tracking
            self.input                  =   {
                "keyboard"          : {}
            }
            self.graphics               =   {
                "GLUint"            :   GLuint()
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
        ####################### 3D ENVIRONEMNT CREATION  #######################
        ########################################################################
            # Create a 3D dict of blocks referenced by xyz coordinate
            self.game_components['blocks'] = {x : {y : {z : Block(Coordinate(x,y,z),"grass") for z in range(self.game_settings["dimension"])} for y in range(1)} for x in range(self.game_settings["dimension"])}
            # Schedule all automatic method calls
            for function_call in self.scheduled_functions:
                pyglet.clock.schedule_interval(function_call, self.scheduled_functions[function_call])
            # Start the world clocks
            self.game_settings['start_time'] = time()
            # Create the list of vertices of the initial blocks
            self.build_vertex_list()
            # Create the VBO
            buffer_data = []
            for three_tuple in self.draw:
                buffer_data.append(three_tuple[0])
                buffer_data.append(three_tuple[1])
                buffer_data.append(three_tuple[2])

            buffer_data = array('f',buffer_data)
            import pprint
            pprint.pp(buffer_data)
            
            buffer_data = (GLfloat * len(buffer_data))(*buffer_data)

            self.buffer = glGenBuffers(1,pointer(self.graphics['GLUint']))

            glBindBuffer(GL_ARRAY_BUFFER, self.graphics['GLUint'])
            glBufferData(GL_ARRAY_BUFFER, len(buffer_data),buffer_data,GL_STATIC_DRAW)
            print("done")




        ########################################################################
        ###################### DECORATED METHODS CREATION  #####################
        ########################################################################
            @self.window.event
            def on_draw():
                # Handle the 3D environment
                self.window.clear()
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(60.0,self.settings['width']/self.settings['height'],self.camera['near'],self.camera['far']);
                gluLookAt(  self.camera['eye']['x'],self.camera['eye']['y'],self.camera['eye']['z'],
                            self.camera['center']['x'],self.camera['center']['y'],self.camera['center']['z'],
                                self.camera['up']['x'],self.camera['up']['y'],self.camera['up']['z'])

                # Handle everything else
                self.movement()
                #self.draw_world()

                ##try:
                #print(f"CAMERA: x:{self.camera['eye']['x']} y: {self.camera['eye']['y']} z: {self.camera['eye']['z']}\nCENTER: x:{self.camera['center']['x']} y: {self.camera['center']['y']} z: {self.camera['center']['z']}\n")
                #print(f"Frametime: {self.game_settings['frame_time']}\nFramerate: {1.0/self.game_settings['frame_time']}\n")
                ##except ZeroDivisionError:
                #glTranslatef(1.0,1.01,1.0)
                print("reached")
                glEnableClientState(GL_VERTEX_ARRAY)
                print("out")
                glVertexPointer(3, GL_FLOAT, 0, 0)
                glDrawArrays(GL_LINES, 0, 3*len(self.draw))

            @self.window.event
            def on_key_press(symbol,modifyer):
                self.input['keyboard'][symbol] = self.game_settings['time']

            @self.window.event
            def on_key_release(symbol,modifyer):
                del self.input['keyboard'][symbol]

            @self.window.event
            def on_mouse_press(x, y, button, modifiers):
                pass

            @self.window.event
            def on_mouse_drag(x, y, dx, dy, button, modifiers):
                pass

            @self.window.event
            def on_mouse_motion(x, y, dx, dy):
                self.camera_vector["angle_horizontal"]  += dx * .01
                if not self.camera_vector["angle_inclination"] <= .01 or not self.camera_vector["angle_inclination"] >= pi - .01 :
                    self.camera_vector["angle_inclination"] -= dy * .01

                self.camera_vector["angle_horizontal"]  = self.camera_vector["angle_horizontal"]        % (2 * pi)

                self.compute_camera_angle()

    def build_vertex_list(self):
        point_pairs = {None}
        final_points_to_draw = {None}
        counter = 0
        for x in self.game_components['blocks']:
            for y in self.game_components['blocks'][x]:
                for z in self.game_components['blocks'][x][y]:
                    block = self.game_components['blocks'][x][y][z]

                    # FOR WIREFRAME
                    block_points = block.Wireframe
                    for i in range(int(len(block_points)/2)):
                        tup1 = block_points[2*i]
                        tup2 = block_points[2*i + 1]
                        package = (tup1, tup2)
                        point_pairs.add(package)

                    #
        final_points_to_draw.remove(None)
        self.draw = []
        point_pairs.remove(None)
        for pair in point_pairs:
            self.draw.append(list(pair[0]))
            self.draw.append(list(pair[1]))

        print(len(self.draw))
        draw_lines(self.draw)

    def run_game(self):
        pyglet.app.run()


    def movement(self):
        movement_step = self.mechanics['player_step']                       + .1 * (LALT in self.input['keyboard'])
        if W in self.input['keyboard']:
            self.camera['eye']['z']     += movement_step     * sin(self.camera_vector['angle_horizontal'])
            self.camera['center']['z']  += movement_step     * sin(self.camera_vector['angle_horizontal'])
            self.camera['eye']['x']     += movement_step     * cos(self.camera_vector['angle_horizontal'])
            self.camera['center']['x']  += movement_step     * cos(self.camera_vector['angle_horizontal'])

        elif S in self.input['keyboard']:
            self.camera['eye']['z']     -= movement_step     * sin(self.camera_vector['angle_horizontal'])
            self.camera['center']['z']  -= movement_step     * sin(self.camera_vector['angle_horizontal'])
            self.camera['eye']['x']     -= movement_step     * cos(self.camera_vector['angle_horizontal'])
            self.camera['center']['x']  -= movement_step     * cos(self.camera_vector['angle_horizontal'])


        if A in self.input['keyboard']:
            self.camera['eye']['z']     -= movement_step     * cos(self.camera_vector['angle_horizontal'])
            self.camera['center']['z']  -= movement_step     * cos(self.camera_vector['angle_horizontal'])
            self.camera['eye']['x']     += movement_step     * sin(self.camera_vector['angle_horizontal'])
            self.camera['center']['x']  += movement_step     * sin(self.camera_vector['angle_horizontal'])
        elif D in self.input['keyboard']:
            self.camera['eye']['z']     += movement_step     * cos(self.camera_vector['angle_horizontal'])
            self.camera['center']['z']  += movement_step     * cos(self.camera_vector['angle_horizontal'])
            self.camera['eye']['x']     -= movement_step     * sin(self.camera_vector['angle_horizontal'])
            self.camera['center']['x']  -= movement_step     * sin(self.camera_vector['angle_horizontal'])

        if SPACE in self.input['keyboard']:
            self.camera['eye']['y'] += movement_step
            self.camera['center']['y'] += movement_step
        elif LSHIFT in self.input['keyboard']:
            self.camera['eye']['y'] -= movement_step
            self.camera['center']['y'] -= movement_step




    # Calculate the position of the "center", or where the camera should
    # Be pointing to
    def compute_camera_angle(self):

        theta   = self.camera_vector["angle_horizontal"]
        phi     = self.camera_vector["angle_inclination"]


        self.camera["center"]["x"] = self.camera['eye']['x'] + 10.0 * sin(phi) * cos(theta)
        self.camera["center"]["z"] = self.camera['eye']['z'] + 10.0 * sin(phi) * sin(theta)


        self.camera['center']['y'] = self.camera['eye']['y'] + 10.0 * cos(phi)

if __name__ == "__main__":
    game = Game()
    game.run_game()

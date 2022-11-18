#local dependencies
from world              import *
from primitives         import draw_line_loop, draw_lines, draw_points, draw_triangle_fan, GL_POINTS, GL_TRIANGLE_FAN
from makeTopoMap        import get_matrix
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
                        GL_ARRAY_BUFFER, GL_STATIC_DRAW, GLfloat, glDrawArrays, glVertexPointer, GL_FLOAT, GL_LINES, GLdouble
from PIL import Image



class Game:
    # Scheduled methods must be declared first
    def update(self,dt):
        pass
    def tick(self, dt):
        self.game_settings['frame_time']    = time() - self.game_settings['time']
        self.game_settings['time'] += self.game_settings['frame_time']


    def __init__(self):
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
                "eye"               :   {'x': 0.0, 'y' : 0.0, 'z' : -10},
                "center"            :   {'x': 0.0, 'y' : 0.0, 'z' : 10.0},
                "up"                :   {'x': 0.0, 'y' : 1.0, 'z' : 0.0},
                "near"              :   1,
                "far"               :   1000
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
                "player_step"       : 1
            }
            # Contains all methods that will be called mapped to their calling interval
            self.scheduled_functions    =   {
                self.update         : 1 / self.mechanics['fps'],
                self.tick           : 1 / self.mechanics['fps']
            }
            # Contains the gameplay component variables
            self.game_settings          =   {
                "dimension"         : 200,
                "time"              : 0.0,
                "start_time"        : 0.0,
                "frame_time"        : 0.0,
                "keyboard"          : {},
                "topo_map"          :None
            }
            # Contains the gameplay items that will be instantiated
            self.game_components        =   {
                "blocks"            :   None
            }
            # Contians all inputs the game is currently tracking
            self.input                  =   {
                "keyboard"          : {}
            }
            # Contians graphical settings
            self.graphics               =   {
                "GLUint"            :   GLuint(),
                "buffer_data"       : [],
                "buffer_size"       : 0
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
            gluPerspective(60.0,self.settings['width']/self.settings['height'],self.camera['near'],self.camera['far'])
            glMatrixMode(GL_MODELVIEW)


        ########################################################################
        ####################### 3D ENVIRONEMNT CREATION  #######################
        ########################################################################
            t1 = time()
            self.game_settings['topo_map'] = get_matrix(seed=int(time()), rows = self.game_settings["dimension"],cols = self.game_settings["dimension"], delta = 1, maxval = 25)
            # Create a 3D dict of blocks referenced by xyz coordinate
            self.game_components['blocks'] = {x : {y : {z : Block(Coordinate(x,self.game_settings['topo_map'][x,z],z),"grass") for z in range(self.game_settings["dimension"])} for y in range(1)} for x in range(self.game_settings["dimension"])}
            # Schedule all automatic method calls
            for function_call in self.scheduled_functions:
                pyglet.clock.schedule_interval(function_call, self.scheduled_functions[function_call])
            # Start the world clocks
            self.game_settings['start_time'] = time()
            #print(f'created in {self.game_settings["start_time"] - t1} seconds')

            print(f'built vertex list')
            # Create the list of vertices of the initial blocks
            self.graphics['buffer_data'] = self.build_vertex_list()
            # Set the size of the buffer, which is 4 bytes PER item
            self.graphics['buffer_size'] = len(self.graphics['buffer_data']) * 4
            # Cast the raw float array into a C-like array
            self.graphics['buffer_data'] = array('f',self.graphics['buffer_data'])
            # Cast the data to a GL-accepted format
            self.graphics['buffer_data'] = (GLfloat * len(self.graphics['buffer_data']))(*self.graphics['buffer_data'])

            # Turn thge buffer on
            glEnableClientState(GL_VERTEX_ARRAY)
            # Create the vertex buffer
            glGenBuffers(1,pointer(self.graphics['GLUint']))
            # Bind the buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.graphics['GLUint'])
            # Fill the data into the buffer
            print(f'loading buffer')

            glBufferData(GL_ARRAY_BUFFER, self.graphics['buffer_size'],self.graphics['buffer_data'],GL_STATIC_DRAW)


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

                # Handle camera and player movement
                self.movement()

                # DEBUG
                print(f"CAMERA: x:{self.camera['eye']['x']} y: {self.camera['eye']['y']} z: {self.camera['eye']['z']}\nCENTER: x:{self.camera['center']['x']} y: {self.camera['center']['y']} z: {self.camera['center']['z']}\n")
                print(f"Frametime: {self.game_settings['frame_time']}\nFramerate: {1.0/self.game_settings['frame_time']}\n")

                # Enable the VBO
                glEnableClientState(GL_VERTEX_ARRAY)
                # Ascribe the right properties
                glVertexPointer(3, GL_FLOAT, 0, 0)
                # Draw the   Buffer
                glDrawArrays(GL_LINES, 0, self.graphics['buffer_size'])

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
        points = []
        i = 1
        for x in self.game_components['blocks']:
            for y in self.game_components['blocks'][x]:
                for z in self.game_components['blocks'][x][y]:
                    block = self.game_components['blocks'][x][y][z]
                    for item in self.wireframe_to_points(block.Wireframe):
                        points.append(item)
            print(f"loaded {i} / {len(self.game_components['blocks'])} chunks")
            i += 1
        return points
    def wireframe_to_points(self,wireframe_list):
        vert_array = []
        for tri_tuple in wireframe_list:
            x,y,z = (tri_tuple[0],tri_tuple[1],tri_tuple[2])
            vert_array.append(x)
            vert_array.append(y)
            vert_array.append(z)
        return vert_array
        
    def run_game(self):
        pyglet.app.run()


    def movement(self):
        movement_step = self.mechanics['player_step'] + .5*(LALT in self.input['keyboard'])

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

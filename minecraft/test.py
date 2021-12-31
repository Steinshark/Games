import pyglet
from pyglet.gl import *

window = pyglet.window.Window()

positionBufferObject = GLuint()
vao = GLuint()

vertexPositions = [0.0, 0.0, 0.0,
                   0.25, 0.0, 0.0,
                   1.75, 1.75, 0.0]

vertexPositionsGl = (GLfloat * len(vertexPositions))(*vertexPositions)

@window.event
def on_draw():
    glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0)
    glDrawArrays(GL_POINTS, 0, 3)
    glDisableVertexAttribArray(0)

glGenBuffers(1, positionBufferObject)
glBindBuffer(GL_ARRAY_BUFFER, positionBufferObject)
glBufferData(GL_ARRAY_BUFFER, len(vertexPositionsGl)*4, vertexPositionsGl, GL_STATIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, 0)

glClearColor(0.0, 0.0, 0.0, 0.0)
pyglet.app.run()

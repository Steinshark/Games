import pyglet
from pyglet.gl import   glBegin, glVertex3f, GL_LINES, GL_POINTS, GL_LINE_LOOP, GL_LINE_STRIP,\
                        GL_TRIANGLES, GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, glEnd
import math
import sys
import numpy

def set_color(colorRGB):
    r,g,b = colorRGB.colors
    glColor3f(r,g,b)

def draw_points(points_list):
        for p in points_list:
            glBegin(GL_POINTS)
            x,y,z = p
            glVertex3f(float(x),float(y),float(z))
            glEnd()
        return

def draw_lines(points_list):
    for i in range(int(len(points_list)/2)):
        p1 = points_list[2*i]
        p2 = points_list[2*i+1]
        glBegin(GL_LINES)
        x,y,z = p1
        glVertex3f(float(x),float(y),float(z))
        x,y,z = p2
        glVertex3f(float(x),float(y),float(z))
        glEnd()
    return

def draw_line_loop(points_list):
    glBegin(GL_LINE_LOOP)
    for v in points_list:
        x,y,z = v
        glVertex3f(float(x),float(y),float(z))
    glEnd()

def draw_line_strip(points_list):
    glBegin(GL_LINE_STRIP)
    for v in points_list:
        x,y,z = v
        glVertex3f(float(x),float(y),float(z))
    glEnd()

def draw_triangle(points_list):
    for i in range(int(len(points_list)/3)):
        p1 = points_list[3*i]
        p1 = points_list[3*i+1]
        p1 = points_list[3*i+2]

        glBegin(GL_TRIANGLES)
        x, y, z = p1
        glVertex3f(x, y, z)
        x, y, z = p2
        glVertex3f(x, y, z)
        x, y, z = p3
        glVertex3f(x, y, z)
        glEnd()
    return

def draw_triangle_strip(points_list):
        glBegin(GL_TRIANGLE_STRIP)
        for v in points_list:
            x,y,z = v
            glVertex3f(float(x),float(y),float(z))
        glEnd()

def draw_triangle_fan(points_list):
        glBegin(GL_TRIANGLE_FAN)
        for v in points_list:
            x,y,z = v
            glVertex3f(float(x),float(y),float(z))
        glEnd()

def WireCube(dim):
    x_min, y_min, z_min = -0.5*dim, -0.5*dim, -0.5*dim
    x_max, y_max, z_max =  0.5*dim,  0.5*dim,  0.5*dim
    glBegin(GL_LINE_STRIP)
    glVertex3f(x_min, y_min, z_min)
    glVertex3f(x_max, y_min, z_min)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_min, y_min, z_min)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_min, z_max)
    glVertex3f(x_min, y_max, z_max)
    glVertex3f(x_min, y_max, z_min)
    glVertex3f(x_max, y_max, z_min)
    glVertex3f(x_max, y_max, z_max)
    glVertex3f(x_max, y_min, z_max)
    glVertex3f(x_max, y_min, z_min)
    glEnd()

def create_circle_list_y(x,y,r,n):
    vertices = []
    for i in range(0,n):
        vertices.append((x+r*numpy.cos((float(i)/n)*2*math.pi),y+r*numpy.sin((float(i)/n)*2*math.pi),0))
    return vertices
def create_circle_list_z(x,z,r,n):
    vertices = []
    for i in range(0,n):
        vertices.append((x+r*numpy.cos((float(i)/n)*2*math.pi),0,z+r*numpy.sin((float(i)/n)*2*math.pi)))
    return vertices

def compute_z(x,y,w,h):
    x_val = math.pow((x - (w/2) ),2)
    y_val = math.pow((y - (h/2) ),2)
    h_val = math.pow((h/2),2)

    expression = h_val- (x_val+y_val)
    check = 0 < expression

    if check:
        return math.sqrt(expression)
    else:
        return 0


def draw_sphere(x,y,z,r,res):
    glBegin(GL_POINTS)
    for x_angle in range(0, 360, res):
        for y_angle in range(0, 180, res):
            # Solve for x, y, z
            x1 = x + r*math.cos(math.radians(y_angle))*math.sin(math.radians(x_angle))
            y1 = y + r*math.sin(math.radians(y_angle))*math.sin(math.radians(x_angle))
            z1 = z + r*math.cos(math.radians(x_angle))
            glVertex3f(x1,y1,z1)
    glEnd()

'''
Created on 28.10.2014

@author: Vladislav
'''

from OpenGL.GL import glClearColor, glMatrixMode, glLoadIdentity, glClear, glOrtho, GL_MODELVIEW, GL_QUADS, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glBegin, glEnd, glFrustum, glColor3f, glTranslatef, glVertex3f, GL_PROJECTION
from OpenGL.GLUT import glutInit, glutInitDisplayMode, glutInitWindowSize, glutCreateWindow, glutDisplayFunc, glutInitWindowPosition
from OpenGL.raw.GLUT import glutMainLoop
from OpenGL.raw.GLUT.constants import GLUT_RGB, GLUT_SINGLE, GLUT_DEPTH

#Global constants
distance = 3.0
focuses = (4.5, 6.0, 7.5)
windows_width = 250
windows_height = 250


def draw_cube_function_0():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_my_cube(focuses[1])

def draw_cube_function_1():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_my_cube(focuses[0])

def draw_cube_function_2():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_my_cube(focuses[1])
    
def draw_cube_function_3():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_my_cube(focuses[2])
    
def draw_my_cube(focus):
    global distance
    glLoadIdentity()
    #Shift the system
    shift_abs = distance + focus
    glTranslatef(0, 0, -shift_abs)
    # Draw Cube that consist of multiple quads
    glBegin(GL_QUADS)
    #Orange ADFE
    glColor3f(1, 0.5, 0)
    glVertex3f(2, -2, 0) #A
    glVertex3f(2, -2, 4) #D 
    glVertex3f(-2, -2, 4) #F
    glVertex3f(-2, -2, 0) #E
    #Green ADCB 
    glColor3f(0, 1, 0)
    glVertex3f(2, -2, 0) #A
    glVertex3f(2, -2, 4) #D 
    glVertex3f(2, 2, 4) #C
    glVertex3f(2, 2, 0) #B 
    #Blue BCGK
    glColor3f(0, 0, 1)
    glVertex3f(2, 2, 0) #B
    glVertex3f(2, 2, 4) #C
    glVertex3f(-2, 2, 4) #G
    glVertex3f(-2, 2, 0) #K 
    #Red GKEF
    glColor3f(1, 0, 0)
    glVertex3f(-2, 2, 4) #G
    glVertex3f(-2, 2, 0) #K 
    glVertex3f(-2, -2, 0) #E
    glVertex3f(-2, -2, 4) #F 
    #White EABK
    glColor3f(1, 1, 1)
    glVertex3f(-2, -2, 0) #E
    glVertex3f(2, -2, 0) #A
    glVertex3f(2, 2, 0) #B
    glVertex3f(-2, 2, 0) #K 
    #Purple FDCG - Take the cap off to see the result
    #glColor3f(1, 0, 1)
    #glVertex3f(-2, -2, 4) #F 
    #glVertex3f(2, -2, 4) #D
    #glVertex3f(2, 2, 4) #C
    #glVertex3f(-2, 2, 4) #G
    glEnd()

def initialize_mode(command, focus):
    global distance
    near = focus
    far = 2 * (near + distance)
    x_min = -5
    x_max = 5
    y_min = -5
    y_max = 5
    glClearColor(0.4, 0.4, 0.4, 0.1)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity() # before Frustum!
    if command == "perspective":
        glFrustum(x_min, x_max, y_min, y_max, near, far)
    elif command == "orthographic":
        glOrtho(x_min, x_max, y_min, y_max, near, far)
    glMatrixMode(GL_MODELVIEW)
    
def main():
    global windows_height
    global windows_width
    windows_x = -windows_width
    windows_y = 0
    windows_x_shift = windows_width + 50
    global focuses
    glutInit() # init the GLUT-machine
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_DEPTH) #RGB scheme, with one buffer, with depth buffer
    #Preparing thr window for orth projection
    glutInitWindowSize(windows_width, windows_height)
    glutInitWindowPosition(windows_x, windows_y)
    glutCreateWindow("Orthographic Projection")
    glutDisplayFunc(draw_cube_function_0)
    initialize_mode("orthographic", focuses[1])
    funcs_for_perspective_drawing = [draw_cube_function_1, draw_cube_function_2, draw_cube_function_3]
    for i in range(len(funcs_for_perspective_drawing)):
        windows_x += windows_x_shift
        glutInitWindowSize(windows_width, windows_height)
        glutInitWindowPosition(windows_x + windows_width, windows_y)
        glutCreateWindow("Persp Proj. Focus = "+str(focuses[i]))
        glutDisplayFunc(funcs_for_perspective_drawing[i])
        initialize_mode("perspective", focuses[i])
    glutMainLoop()

if __name__ == '__main__':
    main()
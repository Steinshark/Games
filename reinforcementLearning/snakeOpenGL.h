    #include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include "snakeBase.h"

#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glu.h>

using namespace std;


////////////////////////////////////////////////////////////////////////
////                              SETTINGS                          ////
//////////////////////////////////////////////////////////////////////// 
int WIDTH   = 800;
int HEIGHT  = 600;
int W_OFFSET = 100;
int H_OFFSET = 100;
const char* WINDOW_NAME = "SnakeAI Viewer";
int TIMER_MSEC_LBOUND = 20;
int GLOBAL_FRAME_COUNT = 0;
int GLOBAL_CTIME = 0;

////////////////////////////////////////////////////////////////////////
////                            draw Objects                        ////
////////////////////////////////////////////////////////////////////////
vector<Coordinate*>     GLOBAL_SNAKE;
Coordinate*             GLOBAL_FOOD;
vector<GLuint>          GLOBAL_VBO_HOLDER; 


////////////////////////////////////////////////////////////////////////
//                          FUNCTION PROTOTYPES                     //// 
////////////////////////////////////////////////////////////////////////
void initOpenGL(char**,int*);
void updateSnake(vector<Coordinate*>);
void upadteFood(Coordinate*);
void draw();
void timer(int);



////////////////////////////////////////////////////////////////////////
//                          FUNCTION DEFINITIONS                    //// 
////////////////////////////////////////////////////////////////////////
void initOpenGL(int* argc,char** argv){
    //init with random ass argc and argv
    glutInit(argc,argv);
    //init display w double buffer, rgba
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    //create window
    glutInitWindowSize(WIDTH,HEIGHT);
    glutInitWindowPosition(W_OFFSET,H_OFFSET);
    glutCreateWindow(WINDOW_NAME);
    //Set the frame updater to be the draw function
    glutDisplayFunc(draw);
    //Set timer func
    glutTimerFunc(TIMER_MSEC_LBOUND,timer,GLOBAL_CTIME);
    //Default clear behavior is black
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    //Start main loop

    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(-1,-1,1,1);
    glutMainLoop();
}


//Setter for the snake
void updateSnake(vector<Coordinate*> newSnake){
    GLOBAL_SNAKE = newSnake;
}


//Setter for the food 
void updateFood(Coordinate* newFood){
    GLOBAL_FOOD = newFood;
}

	
//Draw the snake and food in the window
void draw(){
    cout << "Drawing frame " << GLOBAL_FRAME_COUNT++ << endl;

    glColor3f(1,1,1);
    glBegin(GL_TRIANGLES);
    glVertex2f(1, 0);
    glVertex2f(-1, 0);
    glVertex2f(0, 1);
    glEnd();

    cout << "\tfinished drawing" << endl;
    return;
}


//Helper Function to force screen updates
void timer(int val){   


    glutPostRedisplay();
    glutTimerFunc(TIMER_MSEC_LBOUND,timer,GLOBAL_CTIME);
}

import pygame 
import random
import time 


colors ={"cyan":(0,145,212),"red":(232,56,75),"blue":(45,65,212),"yellow":(250,253,18)}
class hitbox:

    def __init__(self,x,y,dx,color):
        self.x1 = x 
        self.y = y 
        self.x2 = x+dx
        self.color = colors[color]

    def ishit(self,x,y):
        return y == self.y and x >= self.x1 and x < self.x2

class dot:

    def __init__(self,x,y,dx,dy):
        self.x = x 
        self.y = y 
        self.dx = dx 
        self.dy = dy 
    
    def bounce(self,direction,surf):
        if direction == "horizontal":
            self.dy = -1 * self.dy 
        elif direction == "vertical":
            if surf == "paddle":
                self.dx = -1 * self.dx 


def snake():
    screen_w = 600
    screen_h = 600
    tile_width = 10
    tile_height = 10
    red = (255,0,0)
    green = (1,255,40)
    box_width = screen_w / tile_width
    box_height = screen_h / tile_height
    direction = (1,0)

    snake = [(0,0)]
    food = (5,5)

    pygame.init()
    window = pygame.display.set_mode((screen_w,screen_h))

    while True:
        t1 = time.time()
        while time.time() - t1 < .1:
            pygame.event.pump()
        
        keys = pygame.key.get_pressed()
        keys = {(0,-1) : keys[pygame.K_w],
                (-1,0) : keys[pygame.K_a],
                (0,1)  : keys[pygame.K_s],
                (1,0)  : keys[pygame.K_d]}
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()
        
        for dir in keys:
            if keys[dir]:
                direction = dir
        
        window.fill((0,0,0))

        next_head = tuple(map(sum,zip(direction,snake[0])))
        if next_head[0] > tile_width-1 or next_head[0] < 0 or next_head[1] > tile_height-1 or next_head[1] < 0 or next_head in snake:
            print("game over!!!")
            quit()
        elif next_head == food:
            snake = [food] + snake 
            food = (random.randint(0,tile_width-1),random.randint(0,tile_height-1))
            while food in snake:
                food = (random.randint(0,tile_width-1),random.randint(0,tile_height-1))
        else:
            snake = [next_head] + snake[:-1]
        
        for box in snake:
            pygame.draw.rect(window,green,pygame.Rect(box[0]*box_width,box[1]*box_height,box_width,box_height))
        pygame.draw.rect(window,red,pygame.Rect(food[0]*box_width,food[1]*box_height,box_width,box_height))

        pygame.display.flip()

snake()
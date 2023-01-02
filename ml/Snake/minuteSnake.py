import pygame 
import random
import time 

def snake():
    screen_w        = 1000
    screen_h        = 1000
    tile_width      = 20
    tile_height     = 20
    red             = (255,0,0)
    green           = (1,255,40)
    box_width       = screen_w / tile_width
    box_height      = screen_h / tile_height
    direction       = (1,0)

    FRAME_TIME      = .07
    snake           = [(0,0)]
    food            = (5,5)

    pygame.init()
    window = pygame.display.set_mode((screen_w,screen_h))

    while True:
        t1 = time.time()
        while time.time() - t1 < FRAME_TIME:
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
            break
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

while True:
    snake()
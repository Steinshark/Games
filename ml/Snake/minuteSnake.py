import pygame 
import random
import time 
import utilities
import torch 
from matplotlib import pyplot as plt 
import json 
states 		= [] 
actions 	= [] 
eaten 		= [] 
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
	action 			= 0
	FRAME_TIME      = .07
	snake           = [(0,0)]
	food            = (5,5)
	prev_action 	= 0 

	pygame.init()
	window  = pygame.display.set_mode((screen_w,screen_h))
	img     = utilities.build_snake_img(snake,food,(tile_width,tile_height),160,90)
	while True:
		ate 			= 0
		t1 = time.time()
		while time.time() - t1 < FRAME_TIME:
			pygame.event.pump()
			key_pressed = pygame.key.get_pressed()
		if key_pressed[pygame.K_w]:
			action = 0
		elif key_pressed[pygame.K_s]:
			action = 1 
		elif key_pressed[pygame.K_a]:
			action = 2 
		elif key_pressed[pygame.K_d]:
			action = 3 

		keys = {(0,-1) : key_pressed[pygame.K_w],
				(0,1)  : key_pressed[pygame.K_s],
				(-1,0) : key_pressed[pygame.K_a],
				(1,0)  : key_pressed[pygame.K_d]}
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
			snake 	= [food] + snake 
			food 	= (random.randint(0,tile_width-1),random.randint(0,tile_height-1))
			while food in snake:
				food 	= (random.randint(0,tile_width-1),random.randint(0,tile_height-1))
			ate 	= 1 
		else:
			snake = [next_head] + snake[:-1]
		
		for box in snake:
			pygame.draw.rect(window,green,pygame.Rect(box[0]*box_width,box[1]*box_height,box_width,box_height))
		pygame.draw.rect(window,red,pygame.Rect(food[0]*box_width,food[1]*box_height,box_width,box_height))

		pygame.display.flip()
		img     = utilities.step_snake_img(img,snake,food,(tile_width,tile_height),160,90,.33,torch.float16,.1)
		states.append(img)
		actions.append(list(keys.values()))
		eaten.append(ate)

		print(f"action: {action}")
	
	return states,actions,eaten
		
i = 0 
import os
maxn 	= 0  
for file in [f for f in os.listdir('exps') if 'eaten' in f]:
	cur 	= int(file.replace('eaten',""))
	if cur > maxn:
		maxn = cur
i = maxn 

while True:
	utilities.init_utils((20,20),160,90,torch.float16)
	s,a,e = snake()
	name = f"exps/states{i}"
	torch.save(torch.stack(s),name)
	name = f"exps/actions{i}"
	torch.save(torch.tensor(a,dtype=torch.int8),name)
	name = f"exps/eaten{i}"
	torch.save(torch.tensor(e,dtype=torch.int8),name)
	i += 1 

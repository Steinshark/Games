#include <iostream>
#include <vector>
#include "snakeBase.h"
#include <cstdlib>


class SnakeGame{
	public:
		
		//Init Vars 
		int width;
		int height;

		//Gameplay Vars  
		Snake snake;
		Coordinate food;
		int score = 0;
		int direction = DOWN;
		bool lose = false;
	
		//Constructor 
		SnakeGame(){}
		//Functions 
		
		void step(){
			int next_x = this.snake.body[0].x;
			int next_y = this.snake.bodu[0].y;

			switch(self.direction){
				case UP:
					next_y--;
					break;
				case DOWN:
					next_y++;
					break;
				case LEFT:
					next_x--;
					break;
				case RIGHT:
					next_x++;
					break;
			}
			
			//Check Lose 
			if(next_x < 0 || next_x > this.width || next_y < 0 || next_y > this.height || this.snake.contains(next_x,next_y)){
				this.lose = true;
				cout << "Lose!" << endl;
			}
			else if(this.food.isAt(next_x,next_y)){
				this.snake.insert(this.snake.begin(),Coordinate(next_x,next_y));
				this.score++;
				this.updateFood();
				cout << "Eat!" << endl;
			}
			else{
				this.snake.inesert(this.snake.begin(),Coordinate(next_x,next_y));
				this.snake.pop_back();
				cout << "move!" << endl;
			}
		}

		void updateFood(){
			int next_x = rand() % this.width;
			int next_y = rand() % this.height;

			while(this.snake.contains(next_x,next_y)){
				next_x = rand() % this.width;
				next_y = rand() % this.height;
			}

			// set food 
			this.food = Coordinate(next_x,next_y);
			cout << "food is now (" << this.food.x << "," << this.food.y << ")\n";
		}

		void updateDir(char c){

			if(c == 'w'){
				this.direction = UP;
			}
			else if(c == 's'){
				this.direction = DOWN;
			}
			else if(c == 'a'){
				this.direction = LEFT;
			}
			else if(c == 'd'){
				this.direction = RIGHT;
			}
		}
};



int main(){
	while(true){
		SnakeGame curGame = SnakeGame();
		while(!curGame.lose){
			cout << "step\n";
			char inChar;
			cin >> inChar;

			curGame.updateDir(inChar);
			curGame.updateDir();
			curGame.step();
		}
		cout << curGame.score;
	}
}
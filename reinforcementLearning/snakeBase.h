#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

enum Direction{UP,DOWN,LEFT,RIGHT};

struct Coordinate{
	int x,y;

	//Constructor
	Coordinate(int x,int y) {this.x = x;this.y = y;}
	
	//Functions
	boolean isAt(int x,int y){
		return this.x == x && this.y == y;
	}
	boolean operator = (int x,int y){
		return this.x == x && this.y == y;
	}
}

struct Snake{
	vector<Coordinate> body;

	//Constructor 
	Snake(int x, int y){this.body.push_back(Coordinate(x,y));}

	//Functions 
	boolean contains(int x, int y){
		for(iterator ptr = this.body.begin(); ptr != null; ptr++){
			if(this.body[ptr].isAt(x,y)){
				return true;
			}
		}
		return false;
	}

	void step(Direction d){
		int next_x = this.body[0].x;
		int next_y = this.bodu[0].y;

		switch(d){
			case UP:
				this.next_y--;
				break;
			case DOWN:
				this.next_y++;
				break;
			case LEFT:
				this.next_x--;
				break;
			case RIGHT:
				this.next_x++;
				break;
			}
	}
}
		

				



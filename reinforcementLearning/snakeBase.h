#include <iostream>
#include <vector>
#include <math.h>
#include <string>

using namespace std;

enum Direction{UP,DOWN,LEFT,RIGHT};
enum GameState{LOSE,EAT,MOVE};

struct Coordinate{
	int x,y;

	//Constructor
	Coordinate(int x,int y){
		this->x = x;
		this->y = y;
	};
	
	//Functions
	bool isAt(int x,int y){
		return this->x == x && this->y == y;
	}
	bool operator== (Coordinate* c){
		return c->x == this->x && c->y == this->y;
	}
	bool operator== (Coordinate c){
		return c.x == this->x && c.y == this->y;
	}
};

struct Experience{
	vector<float> s;
	char a;
	float r;
	vector<float> s_;
	GameState d;

	Experience(vector<float> s, char a, float r, vector<float> s_, GameState d) : s(s),a(a),r(r),s_(s_),d(d) {}

	void print(){
		//Print out s
		cout << "took action: " << a << endl;
		cout << "got reward: " << r << endl;
		cout << "s----\n";

		for(int i = 0; i < s.size(); i++){
			if(i % (int)(s.size()/3) == 0){
				cout << "\n";
			}
			cout << s[i] << " ";
		}
		cout << "\ns'----\n";
		for(int i = 0; i < s_.size(); i++){
			if(i % (int)(s_.size()/3) == 0){
				cout << "\n";
			}
			cout << s_[i] << " ";
		}
	}
};

struct Snake{
	vector<Coordinate*> body;

	//Constructor 
	Snake(int x, int y){
		this->body.push_back(new Coordinate(x,y));
	};

	//Functions 
	bool contains(int x, int y){

		for(std::vector<Coordinate*>::iterator item = this->body.begin(); item != this->body.end();item++){
			if((**item).isAt(x,y)){
				return true;
			}
		}
		return false;
	}

	void step(Direction d){
		int next_x = this->body[0]->x;
		int next_y = this->body[0]->y;

		switch(d){
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
	}

	void insert(int x, int y){
		this->body.insert(this->body.begin(),new Coordinate(x,y));
	}
	//Overload 
	void print(){
		cout << "[";
		for(std::vector<Coordinate*>::iterator i = this->body.begin(); i != this->body.end();i++){
			cout << "(" << (**i).x << "," << (**i).y  << ") ";
		}
		cout << "]";
	}

};
						
class SnakeGame{
	public:
		
		//Init Vars 
		int width;
		int height;

		//Gameplay Vars  
		Snake* snake;
		Coordinate* food;
		int score;
		int direction;
		bool lose;
	
		//Constructor
		SnakeGame(int x, int y,int width, int height){
			//Game Var inits 
			this-> score = 0;
			this-> lose = false;
			this->width = width;
			this->height = height;

			//Play Var inits 	
			this-> direction = DOWN; 
			this->snake = new Snake(x,y);
			this->updateFood();
		};

		//Functions 
		GameState step(){
			int next_x = this->snake->body[0]->x;
			int next_y = this->snake->body[0]->y;

			switch(this->direction){
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
			if(next_x < 0 || next_x >= this->width || next_y < 0 || next_y >= this->height || this->snake->contains(next_x,next_y)){
				this->lose = true;
				return LOSE;
			}
			else if(this->food->isAt(next_x,next_y)){
				this->snake->body.insert(this->snake->body.begin(),new Coordinate(next_x,next_y));
				this->score++;
				this->updateFood();
				return EAT;
			}
			else{
				this->snake->body.insert(this->snake->body.begin(),new Coordinate(next_x,next_y));
				this->snake->body.pop_back();
				return MOVE;
			}
		}

		void updateFood(){
			int next_x = rand() % this->width;
			int next_y = rand() % this->height;

			while(this->snake->contains(next_x,next_y)){
				next_x = rand() % this->width;
				next_y = rand() % this->height;
			}

			// set food 
			this->food = new Coordinate(next_x,next_y);
		}

		void updateDir(char c){

			if(c == 'w'){
				this->direction = UP;
			}
			else if(c == 's'){
				this->direction = DOWN;
			}
			else if(c == 'a'){
				this->direction = LEFT;
			}
			else if(c == 'd'){
				this->direction = RIGHT;
			}
		}

		//State vector can be of types:
		//	- 3 Channel ({head,body,food})
		//	- 1 Channel (1==head, 2==body,-1 == food)
		vector<float> getStateVector(string encodingType){

			//CH1 == head, CH2 == body, CH3 == food
			if(encodingType == "3Channel"){
				vector<float> v = vector<float>(3*this->height*this->width);
				int channelWidth = (this->width * this->height);
				for(int y = 0; y < this->height;y++){
					for(int x = 0; x < this->width;x++){
						int index = (this->height) * y + x;
						if(this->snake->body.front()->isAt(x,y)){
							v[index + 0*channelWidth] = 1;
							v[index + 1*channelWidth] = 0;
							v[index + 2*channelWidth] = 0;
						}
						else if(this->snake->contains(x,y)){
							v[index + 0*channelWidth] = 0;
							v[index + 1*channelWidth] = 1;
							v[index + 2*channelWidth] = 0;

						}
						else if(this->food->isAt(x,y)){
							v[index + 1*channelWidth] = 0;
							v[index + 0*channelWidth] = 0;
							v[index + 2*channelWidth] = 1; 						}
						else{
							v[index + 0*channelWidth] = 0;
							v[index + 1*channelWidth] = 0;
							v[index + 2*channelWidth] = 0;
						}
					}
				}

				return v;

			}
			else if(encodingType == "1Channel"){
				vector<float> v = vector<float>(this->height*this->width);
				return v;
			}
			else{
				vector<float> v = vector<float>(this->height*this->width);
				return v;
			}


		}

		void printStateVector(){

			//Build vector
			vector<float> v = this->getStateVector("3Channel");

			//Print out vector points
			for(int i = 0; i < v.size(); i++){
				if(i % this->width == 0){
					cout << "\n";
				}

				cout << v[i] << " ";
			}
		}	
};



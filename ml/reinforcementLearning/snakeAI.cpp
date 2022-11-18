#include <iostream>
#include <vector>
#include "snakeOpenGL.h"
#include <cstdlib>
#include <ctime>
#include <cstring>


void runTrainingIterations(int iters){
	vector<char> actions = {'w','s','a','d'};
	srand(time(0));
	vector<Experience> experiences;

	for(int i =0; i < iters; i++){
		SnakeGame game = SnakeGame(0,0,5,5);

		while(!game.lose){
			vector<float> s = game.getStateVector("3Channel");
			char action = actions[rand() % 4];	
			game.updateDir(action);
			GameState d = game.step(); 
			float reward = d == LOSE ? -1 : (d == EAT ? 1 : 0);
			vector<float> s_ = game.getStateVector("3Channel");

			Experience newExp = Experience(s,action,reward,s_,d);
			experiences.push_back(newExp);
		}
	}

	cout << "made " << experiences.size() << "experiences" << endl;
}


int main(int argc,char* argv[]){
	initOpenGL(&argc, argv);

	//temp 

	//height 
	
	}

#include <iostream>
#include <fstream>
using namespace std;


bool inputSource = "interpreted";
double* GLOBAL_TAPE;
int length = 0;
int head = 0;



/*Operations Definition
OP ">"              : = move head right 
OP "<"              : moves head left 
op "?(..)(.)(...)" : cmp .. with item at head. do (.) if true, (...) else 
op "w(..)"          : writes .. to head 
op "+"              : adds *(head) to *(head+1)
op "-"              : subtracts *(head+1) from *(head)
op "while_|...|"    : while head == _, execute |..|
op "!"              : print out the current head as interpreted as bytes
*/   


//char* getNextInput(char* source);

int main(){
    //Create pointers
    int curChar = 255;
    char* intBytePtr = (char*) &curChar;
    cout << "int ptr == " << &curChar << endl << "char ptr == "  << (void*)intBytePtr << "\nval of int is " << curChar << endl;
    cout << "repr of " << curChar << " is size " << sizeof(curChar) << endl;
    
    for(int i = 1; i < 1 + sizeof(curChar);i++){
        cout << "accessing memory location: " << ((&intBytePtr) + i) << " looking ahead "<< sizeof(intBytePtr) << " byte(s), found " <<  hex << (int)(*intBytePtr++) <<  endl;
    }
}


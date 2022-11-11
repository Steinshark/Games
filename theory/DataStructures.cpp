#include <iostream>
#include "lib.h"

using namespace std;


//Used for creating generic types in the data structure 
template <typename T> BTreeNode<T>::BTreeNode(T data){this->data = data;}

// used to create an empty node
template <typename T> BTreeNode<T>::BTreeNode(){;}

//A class used for building binary trees 
template <typename T> class BinaryTree{
    public:
        BTreeNode<T> rootNode;

        BinaryTree(T);
        void add(BTreeNode<T>*);
        void printTree(BTreeNode<T>);
};

//Used to create new tree with a data point
template <typename T> BinaryTree<T>::BinaryTree(T newData){
    BTreeNode<int> temp(newData);
    this->rootNode = temp;
}

//Used to add to tree 
template <typename T> void BinaryTree<T>::add(BTreeNode<T> *newNode){
            BTreeNode<T> *curNode = &rootNode;
            BTreeNode<T> *nextNode;

            while(true){
                //Check end of tree
                if(rootNode.status){
                    this->rootNode = newNode; 
                    this->rootNode->status = true;
                    return;
                }
                //Continue down tree
                else{
                    //Going Right 
                    if(newNode > curNode){
                        //check if right node is NULL 
                        if(!curNode->right->status){
                            curNode->right = newNode;
                            curNode->right->status = true;
                            return;
                        }else{
                            curNode = curNode->right;
                        }
                    }
                    //Going Left
                    else if(newNode < curNode){
                        if(!curNode->left->status){
                            curNode->left = newNode;
                            curNode->left->status = true;
                            return;
                        }else{
                            curNode = curNode->left;
                        }        
                    }
                }
            }
        }




//Used to print tree
template <typename T> void BinaryTree<T>::printTree(BTreeNode<T> b){
            cout << b << " " << printTree(b->left) << " " << printTree(b->left);
        }

int main(){

    BTreeNode<int> a(3);
    BTreeNode<int> b(5);

    cout << "A data is " << a.data << endl;
    cout << "B data is " << b.data << endl;

    
    BinaryTree<int> hyeondong(1);

    hyeondong.add(&b);
}







// Pass by value, pass by reference 
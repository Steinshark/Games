using namespace std;

//Used for creating generic types in the data structure 
template <typename T> 

//A class used for building binary trees 


class BTreeNode{
    public:
        BTreeNode<T> *left;
        BTreeNode<T> *right;

        T data;
        bool status;

        //Constructor
        BTreeNode(T d);
        BTreeNode();

        //For sorting tree
        bool operator > (BTreeNode a){
            cout << "comparing " << this->data << endl;
            return (this->data) > (a.data);
        }
        bool operator < (BTreeNode a){
            cout << "comparing " << this->data << endl;
            return (this->data) < (a.data);
        }
        bool operator == (BTreeNode a){
            cout << "comparing " << this->data << endl;
            return (this->data) == (a.data);
        }
};



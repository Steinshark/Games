#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <ctime>

using namespace std;

int g_len       = 22;
int d_len       = 12;
string FOUT     = "configs.txt";
void print_list(int*,int);

struct config{
    string kernels;
    string strides;
    string padding;
    string in_size;
    string num_channels;
    string outpad;

};

string list_to_string(int* list,int len){
    string s = "[";

    for(int i = 0; i < len; i++){
        s += to_string(list[i]);
        s += ", ";
    }

    s = s.substr(0,s.length()-2) + "]";

    return s;

}

string config_to_string(config c){
    string  s    =  "{\"kernels\":" + c.kernels + ", ";
            s    +=  "\"strides\": " + c.strides + ", ";
            s    +=  "\"paddings\": " + c.padding;

            if(c.outpad.length() > 0){
                s += ", \"out_pad\":" + c.outpad;
            }
            if(c.in_size.length() > 0){
                s += ", \"in_size\":" + c.in_size;
            }
            if(c.num_channels.length() > 0){
                s += ", \"num_channels\":" + c.num_channels;
            }
            s += "}";
    return s;
}

void write_configs(vector<string> v1, vector<string> v2){
    cout << "writing" << endl;
    string s = "{\"D\": [";

    for(auto i = v1.begin(); i < v1.end(); i++){
        s += *i + ", ";
    }
    s = s.substr(0,s.length()-2) + "], \"G\": [";
    for(auto i = v2.begin(); i < v2.end(); i++){
        s += *i + ", ";
    }
    s = s.substr(0,s.length()-2) + "]}";
    
    ofstream configfile;
    configfile.open(FOUT);
    configfile << s;
    configfile.close();

    cout << "wrote to " << FOUT << endl;
}

int out_size_ct(int in,int k,int s,int p,int o){
    return (((in-1)*s) - (2*p) + (k-1) + o + 1); 
}

int total_size_ct(int in,int* kernels,int* strides,int* padding,int* outpad,int* upto){
    int output = in;

    for(int i=0; i < g_len;i++){

        output = out_size_ct(output,kernels[i],strides[i],padding[i],outpad[i]);

        if(output > 5324000){
            return 0;
        }                                 
        if((output < 5324000) && (output >= 5260000)){
            //cout << "cut on " << i << endl;
            //cout << list_to_string(kernels,g_len) << " " << list_to_string(strides,g_len) << " " << list_to_string(padding,g_len) << endl;
            *upto = i+1;
            return 5292000;
        }
    }
    return output;

}

int out_size_c(int in,int k,int s,int p){
    return 1+( (in+(2*p)-(k-1)-1 ) / s);
}

int total_size_c(int in,int* kernels,int* strides,int* padding,bool print,int* upto){
    int output = in;

    for(int i=0; i < d_len;i++){
        
        if (output < kernels[i]){
            return -1;
        }
        
        output = out_size_c(output,kernels[i],strides[i],padding[i]);
        if(print){
            cout << "layer " << i << ": " << output << endl;
        }
        if(output < 0){
            return -1;
        }
        if((output == 1) && (i != (d_len-1))){
            *upto = i+1;
            return 1;
        }
    }
    if(print){
        cout << endl;
    }
    return output;

}

int power_of_2(int low,int high){
    int rand_num = int(rand() % (high-low)) + low;

    if ((low == 0) && (((rand() % (high-low))) == 0)){
        return 0;
    }
    else{
        return pow(2,rand_num);
    }
}

int pure_rand(int low, int high){
    return int(rand() % (high-low)) + low;
}

bool is_sorted_forward(int* list,int len){

    int prev = list[0];

    for(int i = 0; i < len; i++){
        if(!(prev >= list[i])){
            return false;
        }
        prev = list[i];
    } 
    return true;
}

bool is_sorted_backward(int* list,int len){

    int prev = list[0];

    for(int i = 0; i < len; i++){
        if(!(prev <= list[i])){
            return false;
        }
        prev = list[i];
    } 
    return true;
}

bool is_valid(int* list,int len){
    return is_sorted_forward(list,len) || is_sorted_backward(list,len);  
}

void print_list(int* list,int size){
    cout << "list: {";

    for(int i = 0; i < size; i++){
        cout << list[i] <<",";
    }
    cout << "}" << endl;
    int x;
    cin >> x;
}

void fill_list(int* list, int size,int val){
    for(int i = 0; i < size;i++){
        list[i] = val;
    }
}

int main(){

    //START CONFIGS 
    vector<string> possible_d_config;
    vector<string> possible_g_config;
    
    //SET T LIMIT 
    int limit = 60*5;
    cout << "Searching for Discriminator" << endl;

    //Start configs 
    int* kernels = new int[d_len];
    int* strides = new int[d_len];
    int* padding = new int[d_len];
    int uptod    = d_len;
    fill_list(kernels,d_len,2);
    fill_list(strides,d_len,1);
    fill_list(padding,d_len,0);
    time_t start = time(NULL);
    srand(start);

    while((time(NULL)-start) < limit){
        kernels[int(rand()%d_len)] = power_of_2(1,16);
        strides[int(rand()%d_len)] = power_of_2(1,6);
        padding[int(rand()%d_len)] = power_of_2(0,8);
        int out_size = total_size_c(5292000,kernels,strides,padding,false,&uptod);

        
        if ((out_size == 1) && is_sorted_backward(kernels,uptod) && is_valid(strides,uptod) && uptod > 5){
            config dcandidate;
            dcandidate.kernels    = list_to_string(kernels,uptod);
            //dcandidate.num_channels= to_string(uptod);
            dcandidate.strides    = list_to_string(strides,uptod);
            dcandidate.padding    = list_to_string(padding,uptod);
            possible_d_config.push_back(config_to_string(dcandidate));
            //cout << "foundone" <<endl;
        }
    }
    
    cout << "\tFound " << possible_d_config.size() << " configs" << endl;



    cout << "Searching for Generator" << endl;
    int* kernels_g  = new int[g_len];
    int* strides_g  = new int[g_len];
    int* padding_g  = new int[g_len];
    int* out_pad_g  = new int[g_len];

    fill_list(kernels_g,g_len,4);
    fill_list(strides_g,g_len,2);
    fill_list(padding_g,g_len,0);
    fill_list(out_pad_g,g_len,0);

    kernels_g[0] = 4;
    kernels_g[1] = 4;
    kernels_g[2] = 8;
    kernels_g[3] = 16;
    kernels_g[4] = 16;
    int upto = g_len;
    time_t start_g = time(NULL);
    while(time(NULL)-start_g < 0){
        kernels_g[int(rand()%(g_len-5))+5] = power_of_2(4,14);//power_of_2(1,16);
        //padding_g[int(rand()%g_len)] = pure_rand(1,1024*4);
        //strides_g[int(rand()%g_len)] = pure_rand(1,4);//pure_rand(1,10);
        //out_pad_g[int(rand()%g_len)] = pure_rand(1,1024);
        int out_size = total_size_ct(1,kernels_g,strides_g,padding_g,out_pad_g,&upto);

        //cout << out_size << endl;
        if ((out_size == 5292000) && is_sorted_backward(kernels_g,upto) && upto > 5){

            config gcandidate;
            gcandidate.in_size      = to_string(1);
            gcandidate.num_channels = to_string(2);
            gcandidate.kernels      = list_to_string(kernels_g,upto);
            gcandidate.strides      = list_to_string(strides_g,upto);
            gcandidate.padding      = list_to_string(padding_g,upto);
            gcandidate.outpad       = list_to_string(out_pad_g,upto);
            possible_g_config.push_back(config_to_string(gcandidate));
            cout << "Found one" << " size " << to_string(upto) << endl;
        }
    }
    
    cout << "\tFound " << possible_g_config.size() << " configs" << endl;

    write_configs(possible_d_config,possible_g_config);
}
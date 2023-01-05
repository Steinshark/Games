from random import randint
import copy
import time 

kernels_ct = [2] * 4
strides_ct = [1] * 4
padding_ct = [0] * 4

kernels_c = [2] * 8
strides_c = [1] * 8
padding_c = [0] * 8


def out_size_ct(input_size,k,s,p):
    return (input_size-1)*s - 2*p + k-1 + 1


def total_size_ct(input):
    output = input
    for layer in range(len(kernels_ct)):
        output = out_size_ct(output,kernels_ct[layer],strides_ct[layer],padding_ct[layer])
        if output > 5292000:
            return 0

    return output

def out_size_c(input_size,k,s,p):
    return 1 + ((input_size + 2*p - (k-1) - 1) / s) 


def total_size_c(input):
    output = input
    for layer in range(len(kernels_c)):
        output = out_size_c(output,kernels_c[layer],strides_c[layer],padding_c[layer])
        if output < 0:
            return -1

    return output

if __name__ == "__main__":
    goal = 5292000 

    possible_d_config = []
    possible_g_config = []
    t0          = time.time()
    t_thresh    = 5*60 
    
    print("search for Discriminators")
    while time.time()-t0 < t_thresh:
            kernels_c[randint(0,len(kernels_c)-1)] = pow(2,randint(1,14))
            padding_c[randint(0,len(kernels_c)-1)] = pow(2,randint(1,7))
            strides_c[randint(0,len(kernels_c)-1)] = pow(2,randint(0,8))

            #(sorted(kernels_c,reverse=True) == kernels_c)
            outsize = total_size_c(5292000) 
            if outsize < 1024 and outsize > 0 and ((sorted(kernels_c,reverse=True) == kernels_c) or (sorted(kernels_c) == kernels_c)) and ((sorted(strides_c) == strides_c) or (sorted(strides_c,reverse=True) == strides_c)):
                dictionary      = {             'kernels'   :   copy.deepcopy(kernels_c),
                                                'strides'   :   copy.deepcopy(strides_c),
                                                'padding'   :   copy.deepcopy(padding_c)}
                if not dictionary in possible_d_config:
                    possible_d_config.append(dictionary)
                    print("found one")
    t0          = time.time()
    print(f"found {len(possible_d_config)}\n\nsearch for Generators")
    while time.time()-t0 < t_thresh:
        kernels_ct[randint(0,len(kernels_ct)-1)]  = pow(2,randint(1,16))
        padding_ct[randint(0,len(kernels_ct)-1)]  = pow(2,randint(0,8))
        strides_ct[randint(0,len(kernels_ct)-1)]  = randint(1,6)

        if ((total_size_ct(int(44100/4)) == goal) or (total_size_ct(int(44100/2)) == goal) or (total_size_ct(44100) == goal)) and ((sorted(kernels_ct,reverse=True) == kernels_ct) or (sorted(kernels_ct) == kernels_ct)) and ((sorted(strides_ct) == strides_ct) or (sorted(strides_ct,reverse=True) == strides_ct)):
            dictionary  =                   {   'input_size':   total_size_ct(int(44100/4))+total_size_ct(int(44100/2))+total_size_ct(int(44100)),
                                                'kernels'   :   copy.deepcopy(kernels_ct),
                                                'strides'   :   copy.deepcopy(strides_ct),
                                                'padding'   :   copy.deepcopy(padding_ct)}
            if not dictionary in possible_g_config:
                possible_g_config.append(dictionary)
                print("found one")
    print(f"found {len(possible_g_config)}")
    import json 
    f= open("configs.txt","w")
    f.write(json.dumps({'d':possible_d_config,'g':possible_g_config}))
    f.close()



def div_calc(initial,dividend,price,i,max_steps,addl_inv=0):

    if i==max_steps:
        return initial 
    
    else:
        payout = dividend*(initial/price)
        
        return div_calc(initial+payout+addl_inv,dividend,price,i+1,max_steps,addl_inv=addl_inv)



y = 20

print(div_calc(1000,    .14*3,  43.83,  0,  y*4,   addl_inv=4000)*y*1.05, " earned from quarterly")
print(div_calc(1000,    .14,    43.83,  0,  y*12,  addl_inv=1300)*y*1.05, " earned from monthly")
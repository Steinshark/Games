import random 

pool = ["0","1"]
target = "1111111"

sums    = 0 

for i in range(1,1000000):

    st = ""
    while not target in st:
        st = st + random.choice(pool)

    sums += len(st)

    if i % 1000 == 0:
        print(f"iter {i}: avg len {sums/i}") 


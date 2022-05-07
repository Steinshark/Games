import random
from matplotlib import pyplot as plt

def make_string(n):
    s = ''
    for i in range(0,n):
        s += ['0','1','2','3','4','5','6','7','8','9'][random.randint(0,9)]
    return s



title = []
table = [0 for i in range(10,200)]

for l in range(10,200):
    title.append(l)

    origin = make_string(l)
    for i in range(0,100000):
        s = make_string(5)
        if not origin.find(s) == -1:
            table[l-10] += 1


print(title)
print(table)
plt.plot(title,table,marker='o')
plt.show()

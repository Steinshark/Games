import torch 
import time 
a   = torch.randn(size=(3,500,750),dtype=torch.float)

t0  = time.time()
MULT    = 2/255
for _ in range(1000):
    b = a*MULT
    b = b - 1 
print(f"* ran in {(time.time()-t0):.4f}s")

t0  = time.time()
for _ in range(1000):
    with torch.no_grad():
        b = torch.mul(a,MULT) -1 
print(f"mul ran in {(time.time()-t0):.4f}s")

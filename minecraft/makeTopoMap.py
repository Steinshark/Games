#!/usr/bin/python

# Simple library to create a matrix for use with contour mapping (isoline and mesh)
# SI460, Fall AY17, J. Kenney

# get the value at row,col,zv
def get(DB, row, col, *zv):
    if not zv:
        if row in DB and col in DB[row]:
            return DB[row][col]
        else:
            return -1
    if row in DB and col in DB[row] and zv[0] in DB[row][col]:
        return DB[row][col][zv[0]]
    return -1

# set the value at row,col,*zv, val
def set(DB, row, col, *n):
    val = n[-1]
    zv = -1
    if len(n) >= 2:
        zv = n[0]
    if not row in DB:
        DB[row] = {}
    if zv != -1 and not col in DB[row]:
        DB[row][col] = {}
    if zv == -1:
        DB[row][col] = val
    else:
        DB[row][col][zv] = val

# Rotate a 2D matrix
def mod_matrix(M):
    tmp = M[0]
    for row in range(len(M)-1):
        M[row] = M[row+1]
    M[len(M)-1] = tmp
    return M

# Randomize the existing matrix
def mod_matrix2(M, delta=1, maxval=20):
    import random
    for row in range(len(M)):
        for col in range(len(M[0])):
            vals = []
            try:
                vals.append(M[row-1][col])
            except:
                pass
            try:
                vals.append(M[row+1][col])
            except:
                pass
            try:
                vals.append(M[row][col-1])
            except:
                pass
            try:
                vals.append(M[row][col+1])
            except:
                pass
            mv = max(min(vals)-delta,0)
            Mv = min(max(vals)+delta,maxval)
            nv = random.randint(mv,Mv)
            M[row][col] = nv
    return M

# Build a 3D space for use with
def get_3dmatrix(seed=42, rows=40, cols=40, zv=40, delta=3, maxval=20):
    import random, numpy
    random.seed(seed)
    DB = {}
    for row in range(rows):
        for col in range(cols):
            for z in range(zv):
                up = get(DB, row-1, col, z)
                lf = get(DB, row, col-1, z)
                bk = get(DB, row, col, z-1)
                cc = []
                mh = []
                ml = []
                if up != -1:
                    cc.append(up)
                    mh.append(up+delta)
                    ml.append(max(0, up-delta))
                if lf != -1:
                    cc.append(lf)
                    mh.append(lf+delta)
                    ml.append(max(0, lf-delta))
                if bk != -1:
                    cc.append(bk)
                    mh.append(bk+delta)
                    ml.append(max(0, bk-delta))
                if up == -1 and lf == -1 and bk == -1:
                    val = 0
                else:
                    if max(ml) == min(mh):
                        val = max(ml)
                    elif max(ml) > min(mh):
                        val = random.randint(min(mh), max(ml))
                    else:
                        val = random.randint(max(ml), min(mh))
                if val > maxval:
                    val = maxval
                set(DB, row, col, z, val)
    M = []
    for row in range(rows):
        t = []
        for col in range(cols):
            t2 = []
            for z in range(zv):
                t2.append(get(DB,row,col,z))
            t.append(t2)
        M.append(t)
    return numpy.array(M)

# Build a 2D space for use with marching squares
def get_matrix(seed=42, rows=400, cols=400, delta=3, maxval=20):
    import random, numpy
    random.seed(seed)
    DB = {}
    for row in range(rows):
        for col in range(cols):
            up = get(DB, row-1, col)
            lf = get(DB, row, col-1)
            if up == -1 and lf == -1:
                val = random.randint(1,10)
                val = 0
            elif up != -1 and lf == -1:
                val = random.randint(max(0, up-delta), up+delta)
            elif up == -1 and lf != -1:
                val = random.randint(max(0, lf-delta), lf+delta)
            else:
                if max(0, up-delta, lf-delta) == min(lf+delta, up+delta):
                    val = max(0, up-delta, lf-delta)
                elif max(0, up-delta, lf-delta) > min(lf+delta, up+delta):
                    val = random.randint(min(lf+delta, up+delta), max(0, up-delta, lf-delta))
                else:
                    val = random.randint(max(0, up-delta, lf-delta), min(lf+delta, up+delta))
            if val > maxval:
                val = maxval
            set(DB, row, col, val)
    M = []
    for row in range(rows):
        t = []
        for col in range(cols):
            t.append(get(DB,row,col))
        M.append(t)
    return numpy.array(M)

if __name__ == '__main__':
    M = get_matrix(rows=10, cols=10)
    M3 = get_3dmatrix(rows=10, cols=10, zv=10)

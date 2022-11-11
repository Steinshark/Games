#Reverse polish notation calc 
from pprint import pp

DEBUG = False

ops = {
    "mul" : lambda x : multiplication(x),
    "div" : lambda x : execute(x[0])/execute(x[1]),
    "+" : lambda x : addition(x),
    "-" : lambda x,y : float(x)-float(y),
    "set" : lambda x : setVal(x),
    "map" : lambda x : mapExec(x),
    "list" : lambda x : listExec(x),
    "len" : lambda x : lenExec(x),
    "define" : lambda x : defineExec(x) and print(f"define call given {x}") or print(f"called{x[0].split('(')[1].split(' ')[0]}")
}

def getVal(expr):
    try:
        return float(expr)
    except ValueError:
        try:
            return vars[expr]
        except KeyError:
            return expr

def setVal(x):
    if DEBUG:
        print(f"setting {x}")
    key,value = x[0], x[1]
    ops[key] = value
    pp(ops)
    return value

def addition(argsList):

    if DEBUG:
        print(f"\tDEBUG: addition with {argsList}")
    if len(argsList) == 0:
        return 0

    if len(argsList) == 1:
        return execute(argsList[0])

    argsList = [execute(item) for item in argsList]
    while not len(argsList) == 1:
        argsList[1] = argsList[0] + argsList[1]
        if DEBUG:
            print(f"adding {argsList[0]} by {argsList[1]}")
        argsList.pop(0)

    return argsList[0]

def multiplication(argsList):
    if len(argsList) == 0:
        return 0
    
    argsList = [execute(item) for item in argsList]
    while not len(argsList) == 1:
        argsList[1] = argsList[0] * argsList[1]
        if DEBUG:
            print(f"multiplying {argsList[0]} by {argsList[1]}")
        argsList.pop(0)

    return argsList[0]

def createFuncEnv(argList,valList):
    if not len(argList) == len(valList):
        print(f"bad input given to fun w arg: {argList} and val: {valList}")
        return False
    else:
        for var,val in zip(argList,valList):
            ops[var] = val
    pp("ops now")
    pp(ops)
    return 0


def callFun(args,x,expr):
    if DEBUG:
        print(f"calling fun with {args} and {x}\n on {expr}")
    createFuncEnv(args,x) 
    res = execute(expr)
    print(f"resulted in {res}")
    return execute(expr)

def defineExec(statement):
    name = statement[0][1:-1].split(" ")[0]
    args = statement[0][1:-1].split(" ")[1:]
    expr = statement[1]
    #ops[name] = lambda x: createFuncEnv(args,x) and execute(expr)
    ops[name] = lambda x: callFun(args,x,expr)

    if DEBUG:
        print(f"op: {name} and args {args} original {statement} exec: {expr}")
        pp(ops)

def listExec(argsList):
    if DEBUG:
        print(f"returning list of {argsList}")
    return argsList

def mapExec(argsList):
    op = argsList[0]
    operands = argsList[1]

    if DEBUG:
        print(f"calling op: {op} on list: {operands}")
    return ops[op](execute(operands))

def lenExec(argsList):
    if DEBUG:
        print(f"recieved {argsList}")
    return len(execute(argsList[0]))

def execute(expr_str):
    expr_str = expr_str.strip()
    #Either an atom or an expression
    if(not expr_str[0] == "("):
        #Try VAR, then FLOAT 
        try:
            return execute(ops[expr_str])
        except KeyError:
            try:
                return float(expr_str)
            except ValueError:
                pass
    else:   
        expressions = []
        expr = expr_str[1:-1]
        op = expr.split(" ")[0].strip()
        par_expr = False
        expressions = []
        #Get sub exprs
        expr = expr[expr.find(" "):]

        for i,c in enumerate(expr):
            #If par, its either a new expression or nested expr
            if c == "(":
                # If new expression, append 
                if not par_expr:
                    expressions.append(c)
                # If nested Expr, continue adding normalling, but incr open count
                else:
                    expressions[-1] += c 
                par_expr += 1
            elif c == ")":
                expressions[-1] += c
                par_expr -= 1
            # If " ", either a new expr or nested expr
            elif c == " ":
                if par_expr:
                    expressions[-1] += c 
                elif expr[1+i] == "(":
                    continue 
                else:
                    expressions.append(c)
            else:
                if len(expressions) == 0:
                    expressions.append(c)
                else:
                    expressions[-1] += c

        expressions = [op] + [e.strip() for e in expressions]

        if DEBUG:
            print(f"op: {op}")
            print(f"\tDEBUG:evaluated to {expressions}")

        return ops[expressions[0]](expressions[1:])

if __name__ == "__main__":
    line = input("user in:> ") 
    while not line == "(quit)":
        print(execute(line))
        line = input("user in:> ") 
#Reverse polish notation calc 
from pprint import pp

DEBUG = False

ops = {
    "mul" : lambda x : multiplication(x),
    "div" : lambda x,y : float(x)/float(y),
    "+" : lambda x : addition(x),
    "-" : lambda x,y : float(x)-float(y),
    "set" : lambda x : setVal(x),
    "define" : lambda x : print(f"define call given {split_expr(x)}") or print(f"called{x[0].split('(')[1].split(' ')[0]}")
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
    print(f"setting {x}")
    key,value = x[0], x[1]
    ops[key] = value
    pp(ops)
    return value

def addition(argsList):

    print(f"\tDEBUG: addition with {argsList}")
    if len(argsList) == 0:
        return 0

    if len(argsList) == 1:
        return execute(argsList[0])

    while not len(argsList) == 1:
        argsList[1] = execute(argsList[0]) + execute(argsList[1])
        argsList.pop(0)

    return argsList[0]

def multiplication(argsList):
    if len(argsList) == 0:
        return 0

    while not len(argsList) == 1:
        argsList[1] = execute(argsList[0]) * execute(argsList[1])
        argsList.pop(0)
    return argsList[0]

def defineExec(args):
    name = args[0].split('(')[1].split(' ')[0]
    args = None

def split_expr(expr):
    expr = expr.strip()
    if not expr[0] == "(":
        return ops[expr]
    #Get op 
    op = expr.split("(")[1].split(" ")[0]

    par_expr = False
    expressions = []
    #Get sub exprs
    expr = expr[expr.find(" "):-1]
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

    expressions = [e.strip() for e in expressions]
    print(f"ended with op: {op} and exprs: {expressions}")
    return [op] + expressions

def execute(expr_str):
    expr_str = expr_str.strip()
    #Either an atom or an expression
    if(not expr_str[0] == "("):
        #Try VAR, then FLOAT 
        try:
            return execute(ops[expr_str])
        except KeyError:
            return float(expr_str)
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

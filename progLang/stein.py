#Reverse polish notation calc 

vars = {}

def getVal(expr):
    try:
        return float(expr)
    except ValueError:
        try:
            return vars[expr]
        except KeyError:
            return expr

def setVal(x):
    key,value = x[0], x[1]
    vars[key] = getVal(value)
    print(vars)
    return value



def addition(argsList):

    if len(argsList) == 0:
        return 0

    while not len(argsList) == 1:
        argsList[1] = getVal(argsList[0]) + getVal(argsList[1])
        argsList.pop(0)
    return argsList[0]

def multiplication(argsList):
    if len(argsList) == 0:
        return 0

    while not len(argsList) == 1:
        argsList[1] = getVal(argsList[0]) * getVal(argsList[1])
        argsList.pop(0)
    return argsList[0]

ops = {
    "x" : lambda x : multiplication(x),
    "div" : lambda x,y : float(x)/float(y),
    "+" : lambda x : addition(x),
    "-" : lambda x,y : float(x)-float(y),
    "set" : lambda x : setVal(x)
}





#Legal expr always in form: (op expr1 ... expr2 ... exprn) || x x 3 {R} (x element of reals)

def eval(expr):

    expr = expr.strip()

    #Either return a constant  
    if not "(" in expr:
        return getVal(expr)

    #Or evaluate an operation
    else:
        op = expr.split("(")[1].split(" ")[0]
        #ensure valid op 
        if not op in ops.keys():
            return

        #evaluate sub expressions
        
        expr_str = " ".join(expr.split(" ")[1:])[:-1]

        expressions = [] 
        par_expr = 0 
        for i,c in enumerate(expr_str):
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
                elif expr_str[1+i] == "(":
                    continue 
                else:
                    expressions.append(c)
            else:
                if len(expressions) == 0:
                    expressions.append(c)
                else:
                    expressions[-1] += c

        expressions = [e.strip() for e in expressions]
        return ops[op](expressions)




# (op x x )
if __name__ == "__main__":
    line = "go ahead!"
    while not line == "(quit)":
        line = input("user in:> ") 

        print(eval(line))

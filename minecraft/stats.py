

file = open("out.txt")
flag = False

full_list = []
for line in file.readlines():

    try:
        if flag:
            line_elements = [x for x in line.split(" ") if not x == '']

            stats = {   "calls: "       :   int(line_elements[0].split('/')[0]),
                        "total: "       :   float(line_elements[1]),
                        "function: "    : line_elements[-1]}

            if stats['total: '] > 1:
                print(f"{stats['function: ']}: \t\t{stats['total: ']}")

            full_list.append(stats)
    except ValueError:
        pass
    if line.find("ncalls") > 0:
        flag = True

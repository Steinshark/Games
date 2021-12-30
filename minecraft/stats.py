#
#
#file = open("out.txt")
#flag = False
#
#full_list = []
#for line in file.readlines():
#
#    try:
#        if flag:
#            line_elements = [x for x in line.split(" ") if not x == '']
#
#            stats = {   "calls: "       :   int(line_elements[0].split('/')[0]),
#                        "total: "       :   float(line_elements[1]),
#                        "function: "    : line_elements[-1]}
#
#            if stats['total: '] > 1:
#                print(f"{stats['function: ']}: \t\t{stats['total: ']}")
#
#            full_list.append(stats)
#    except ValueError:
#        pass
#    if line.find("ncalls") > 0:
#        flag = True



#import pyglet.gl
#out = dir(pyglet.gl)
#for item in out:
#    if item.find("ERROR") > 0:
#        print(item)


vals = [
    "=Rotation!$D23:$D25",
    "=Rotation!$D34:$D36",
    "=Rotation!$D44:$D47",
    "=Rotation!$K11:$K14",
    "=Rotation!$K22:$K25",
    "=Rotation!$K33:$K36",
    "=Rotation!$K45:$K47",
    "=Rotation!$R11:$R14",
    "=Rotation!$R22:$R25",
    "=Rotation!$R44:$R47",
    "=Rotation!$Y11:$Y14",
    "=Rotation!$Y22:$Y25",
    "=Rotation!$Y33:$Y36",
    "=Rotation!$Y44:$Y47",
    "=Rotation!$AF11:$AF14",
    "=Rotation!$AF22:$AF25",
    "=Rotation!$AF33:$AF36",
    "=Rotation!$AF44:$AF47",
    "=Rotation!$AM11:$AM14",

    $D$15:$D$17;$D$26:$D$28;$D$37:$D$39;$D$48:$D$50;$K$15:$K$17;$K$26:$K$28;$K$37:$K$39;$K$48:$K$50;$R$15:$R$17;$R$48:$R$50;$Y$15:$Y$17;$Y$26:$Y$28;$Y$37:$Y$39;$Y$48:$Y$50;$AF$15:$AF$17;$AF$26:$AF$28;$AF$37:$AF$39;$AF$48:$AF$50;$AM$15:$AM$17;$R$26
]
import string
file = open("out",'w')
for entry in vals:
    range_start = int(entry.split("$")[1][-3:].split(':')[0])
    range_end   = int(entry.split("$")[2][-2:])

    col =  entry.split("$")[1][0]
    if entry.split("$")[1][1] in string.ascii_uppercase:
        col += entry.split("$")[1][1]
    for i in range(range_start,range_end+1):
        file.write("=Rotation!" + str(col) + str(i) + '\n')

import logging
import os.path
import sys

import codecs

c = 0 #right
e = 0 #wrong
n = 0 #the number of words in gold_standard
debug = 0 #the number of words in gold_standard

gold = "../icwb2-data/gold/as_testing_gold.utf8"
test = "result.utf8"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

finput = codecs.open(gold, encoding='utf-8')
foutput = codecs.open(test, encoding='utf-8')
finput_line = finput.readlines()
foutput_line = foutput.readlines()


for index in range(len(finput_line)):
    list1 = []
    list2 = []
    i = 1
    start = 1
    end = 1

    for word in finput_line[index]:
        if word == '　':
            end = i-1
            list1.append((start, end))
            start = i
        else:
            i += 1

    i = 1
    start = 1
    end = 1

    for word in foutput_line[index]:
        if word == ' ':
            end = i-1
            list2.append((start, end))
            start = i
        else:
            i += 1
    if(len(list2)>0):
        list2.pop()

    i = 0
    j = 0
    n += len(list1) # update n
    debug += len(list2)

    print(list1)
    print(list2)

    while ((i<len(list1)) and (j<len(list2))):
        if(list1[i][0] <= list2[j][0] and list1[i][1] > list2[j][0]): # in the range
            if(list1[i][0] == list2[j][0]):
                if(list1[i][1] == list2[j][1]):
                    c += 1
                    i += 1
                    j += 1
                    break
                else:
                    e += 1
                    j += 1
                    break
            else:
                e += 1
                j += 1
                break
        else:
            # if(list1[i][0] > list2[j][0]):
            #     print("wrong")
            i += 1
            break

    if(j<len(list2)):
        e += len(list2)-j

print(n)
print(e)
print(c)
print(debug)

R = c/n
P = c/(c+e)
F = 2*P*R/(P+R)
ER = e/n

if(debug != c+e):
    print("bug!")

print("Recall:",end='')
print(R)
print("Pecision:",end='')
print(P)
print("F-measure:",end='')
print(F)
print("Error Rate:",end='')
print(ER)


finput.close()
foutput.close()
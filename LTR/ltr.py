#!/bin/python3
import numpy as np

a = np.array([[-7,-7],[1,1],[3,3],[5,5],[7,7]])
b = np.array([[-5,-4],[-2,-1],[3,4],[5,6],[8,9]])

ws = np.random.randn(2)
gap = 1.0
rate = 0.0005

flag = 1
loop = 1
while(flag):
    flag = 0
    s = np.zeros(2)
    num = 0
    for i in range(len(a)):
        for j in range(len(b)):
            xa = a[i]
            xb = b[j]
            va = sum(xa * ws)
            vb = sum(xb * ws)
            if (va < vb):
                flag = 1
                ya = vb + gap
                yb = va - gap
                num = num + 2
                s += (ws * xa - ya) * xa + (ws * xb - yb) * xb
    if(flag):
        ws = ws - rate * s/num
    if(loop % 100 == 0):
        print(ws)
    loop = loop + 1

print("迭代次数：",loop)
print("回归系数：",ws)


va = np.sum(a * ws,1)
vb = np.sum(b * ws,1)
print("回归结果：",va,vb)
print("对比差值：",va-vb)
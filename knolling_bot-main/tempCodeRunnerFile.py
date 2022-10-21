import numpy as np
  #获取当前程序段中的全体局部变量名  

num_cube = 12
fac = [] # 定义一个列表存放因子
for i in range(2, num_cube): # 这里的逻辑和你一样
    if num_cube % i == 0:
        fac.append(i) # 如果是因子就放进去
        continue
    else:
        pass
print(fac)
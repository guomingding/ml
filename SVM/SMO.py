C = 1
threshold = 1e-8  # 确保选择的a2使得损失函数有足够的下降
eps = 1e-30


# 参数顺序：K,b,alpha,y
def getL(K, alpha, y):  # 获取当前损失函数值
    l = 0
    for i in range(len(y)):
        for j in range(len(y)):
            l += 0.5 * alpha[i] * alpha[j] * y[i] * y[j] * K[i][j]
        l -= alpha[i]
    return l


def getGx(K, b, i, alpha, y):  # 获取预测值
    g = b
    for j in range(len(y)):
        g += alpha[j] * y[j] * K[j][i]
    return g


# 不要直接修改alpha，防止不满足目标函数阈值下降条件的情况出现
def getNewAlpha(K, b, i1, i2, alpha, y):
    eta = K[i1][i1] + K[i2][i2] - 2 * K[i1][i2]
    # ************
    # if eta < 1e-20:
    #     eta = 1e-20
    # ************
    a2_new_unc = alpha[i2] + y[i2] * ((getGx(K, b, i1, alpha, y) - y[i1]) - (getGx(K, b, i2, alpha, y) - y[i2])) / eta
    # 剪辑
    if y[i1] == y[i2]:
        L = max(0, alpha[i1] + alpha[i2] - C)
        H = min(C, alpha[i1] + alpha[i2])
    else:
        L = max(0, alpha[i2] - alpha[i1])
        H = min(C, C + alpha[i2] - alpha[i1])
    if a2_new_unc > H:
        a2_new = H
    elif a2_new_unc < L:
        a2_new = L
    else:
        a2_new = a2_new_unc

    a1_new = alpha[i1] + y[i1] * y[i2] * (alpha[i2] - a2_new)
    # a1 = y[i1] * st - a2_new * y[i1] * y[i2]

    E1 = getGx(K, b, i1, alpha, y) - y[i1]
    E2 = getGx(K, b, i2, alpha, y) - y[i2]
    b1_new = -1 * E1 - y[i1] * K[i1][i1] * (a1_new - alpha[i1]) - y[i2] * K[i2][i1] * (a2_new - alpha[i2]) + b
    b2_new = -1 * E2 - y[i1] * K[i1][i2] * (a1_new - alpha[i1]) - y[i2] * K[i2][i2] * (a2_new - alpha[i2]) + b
    if (a1_new > 0 and a1_new < C) and (a2_new and a2_new < C):
        bnew = b1_new
    else:
        bnew = (b1_new + b2_new) / 2
    return (a1_new, a2_new, bnew)


def SMO(K, b, alpha, y):
    flag = True
    while flag == True:
        # flag = false说明找不到可以优化的a1,a2
        flag = False
        preL = getL(K, alpha, y)
        for i1 in range(len(alpha)):
            gxi1 = getGx(K, b, i1, alpha, y)
            if alpha[i1] > 0 and alpha[i1] < C and abs(y[i1] * gxi1 - 1) > eps:
                E1 = gxi1 - y[i1]
                maxDiff = 0  # |使得E1 - E2|最大
                posi2 = 0
                for i2 in range(len(alpha)):
                    if i1 == i2:
                        continue
                    E2 = getGx(K, b, i2, alpha, y) - y[i2]
                    if abs(E1 - E2) > maxDiff:
                        maxDiff = abs(E1 - E2)
                        posi2 = i2

                a1_new, a2_new, b_new = getNewAlpha(K, b, i1, posi2, alpha, y)
                # 更新alpha
                alpha[i1], a1_new = a1_new, alpha[i1]
                alpha[posi2], a2_new = a2_new, alpha[posi2]
                curL = getL(K, alpha, y)

                # 找到解，更新b
                if preL - curL >= threshold:
                    b = b_new
                    flag = True
                    break
                # 没有达到预定条件,那么搜索在0-C内的a2
                # if preL - curL < threshold:
                else:
                    alpha[i1], a1_new = a1_new, alpha[i1]
                    alpha[posi2], a2_new = a2_new, alpha[posi2]
                    for i2 in range(len(alpha)):
                        if i2 == i1 or i2 == posi2:
                            continue
                        if alpha[i2] > 0 and alpha[i2] < C:
                            # i2写成了posi2
                            a1_new, a2_new, b_new = getNewAlpha(K, b, i1, i2, alpha, y)
                            alpha[i1], a1_new = a1_new, alpha[i1]
                            alpha[i2], a2_new = a2_new, alpha[i2]
                            curL = getL(K, alpha, y)
                            # 满足条件则停止
                            if preL - curL >= threshold:
                                b = b_new
                                flag = True
                                break
                            else:
                                alpha[i1], a1_new = a1_new, alpha[i1]
                                alpha[i2], a2_new = a2_new, alpha[i2]
                    if flag:
                        break
                    # 如果还是不满足，就搜索全部的a2
                    else:
                        for i2 in range(len(alpha)):
                            if i2 == i1 or i2 == posi2:
                                continue
                            if abs(alpha[i2] - 0) <= eps or abs(alpha[i2] - C) <= eps:
                                a1_new, a2_new, b_new = getNewAlpha(K, b, i1, i2, alpha, y)
                                alpha[i1], a1_new = a1_new, alpha[i1]
                                alpha[i2], a2_new = a2_new, alpha[i2]
                                curL = getL(K, alpha, y)

                                if preL - curL >= threshold:
                                    b = b_new
                                    flag = True
                                    break
                                else:
                                    alpha[i1], a1_new = a1_new, alpha[i1]
                                    alpha[i2], a2_new = a2_new, alpha[i2]
            if flag:
                break

        if flag == False:
            for i1 in range(len(alpha)):
                gxi1 = getGx(K, b, i1, alpha, y)
                if ((alpha[i1] - 0) <= eps and y[i1] * gxi1 < 1) or ((alpha[i1] - C) < eps and y[i1] * gxi1 > 1):
                    E1 = gxi1 - y[i1]
                    maxDiff = 0  # |使得E1 - E2|最大
                    posi2 = 0
                    for i2 in range(len(alpha)):
                        if i2 == i1:
                            continue
                        E2 = getGx(K, b, i2, alpha, y) - y[i2]
                        if abs(E1 - E2) > maxDiff:
                            maxDiff = abs(E1 - E2)
                            posi2 = i2

                    a1_new, a2_new, b_new = getNewAlpha(K, b, i1, posi2, alpha, y)
                    # 更新alpha
                    alpha[i1], a1_new = a1_new, alpha[i1]
                    alpha[posi2], a2_new = a2_new, alpha[posi2]
                    curL = getL(K, alpha, y)

                    if preL - curL >= threshold:
                        b = b_new
                        flag = True
                        break

                    # 没有达到预定条件,那么搜索在0-C内的a2
                    # if preL - curL < threshold:
                    else:
                        alpha[i1], a1_new = a1_new, alpha[i1]
                        alpha[posi2], a2_new = a2_new, alpha[posi2]

                        for i2 in range(len(alpha)):
                            if i2 == i1 or i2 == posi2:
                                continue
                            if alpha[i2] > 0 and alpha[i2] < C:
                                a1_new, a2_new, b_new = getNewAlpha(K, b, i1, i2, alpha, y)
                                alpha[i1], a1_new = a1_new, alpha[i1]
                                alpha[i2], a2_new = a2_new, alpha[i2]
                                curL = getL(K, alpha, y)
                                # 满足条件则停止
                                if preL - curL >= threshold:
                                    b = b_new
                                    flag = True
                                    break
                                else:
                                    alpha[i1], a1_new = a1_new, alpha[i1]
                                    alpha[i2], a2_new = a2_new, alpha[i2]
                        if flag:
                            break
                        # 如果还是不满足，就搜索全部的a2
                        else:
                            for i2 in range(len(alpha)):
                                if i2 == i1 or i2 == posi2:
                                    continue
                                if (alpha[i2] - 0) <= eps or (alpha[i2] - C) <= eps:
                                    a1_new, a2_new, b_new = getNewAlpha(K, b, i1, i2, alpha, y)
                                    alpha[i1], a1_new = a1_new, alpha[i1]
                                    alpha[i2], a2_new = a2_new, alpha[i2]
                                    curL = getL(K, alpha, y)

                                    if preL - curL >= threshold:
                                        b = b_new
                                        flag = True
                                        break
                                    else:
                                        alpha[i1], a1_new = a1_new, alpha[i1]
                                        alpha[i2], a2_new = a2_new, alpha[i2]
                if flag:
                    break

    return alpha, b


def getK(X, f):
    K = [0] * len(X)
    for i in range(len(K)):
        K[i] = [0] * len(X)
    # print(len(K))
    # print(len(K[0]))
    for i in range(len(X)):
        for j in range(len(X)):
            for k in range(len(X[i])):
                K[i][j] += f(X[i][k]) * f(X[j][k])
    return K


# X = [[3,3],[4,3],[1,1]]
# y = [1,1,-1]
# K = getK(X,lambda x:x)
# alpha = [0,0,0]
# b = 0
# alpha,b = SMO(K,b,alpha,y)
# print(alpha,b)

import random
import numpy as np
import matplotlib.pyplot as plt


def kernelF(x, xi, f):
    Kx_xi = 0
    for i in range(len(x)):
        Kx_xi += f(x[i]) * f(xi[i])
    return Kx_xi


#定义椭圆x^2 + 4y^2 = 4
# 圆圈是训练集数据，三角形是测试集数据，用颜色标记正负例
# theta = np.arange(0, 2.1 * np.pi ,0.1 * np.pi)
# plt.plot(2 * np.cos(theta),np.sin(theta),c='red')
#
# X = [[random.uniform(-2.5,2.5),random.uniform(-1.5,1.5)] for _ in range(50)]
# y = [0] * len(X)
# for i in range(len(X)):
#     if X[i][0] ** 2 + 4 * X[i][1] ** 2 <= 4:
#         y[i] = 1
#         plt.scatter(X[i][0],X[i][1],c='blue',marker='o')
#     else:
#         y[i] = -1
#         plt.scatter(X[i][0], X[i][1], c='green',marker='o')
#
# K = getK(X,lambda x:x ** 2)
# b = 0
# alpha = [0] * len(X)
# alpha,b = SMO(K,b,alpha,y)
#
# for i in range(50):
#     xi = [random.uniform(-2.5,2.5),random.uniform(-1.5,1.5)]
#     pred = b
#     for idx in range(len(alpha)):
#         pred += alpha[idx] * y[idx] * kernelF(xi,X[idx],lambda x:x ** 2)
#     if pred >= 0:
#         plt.scatter(xi[0], xi[1], c='blue', marker='^')
#     else:
#         plt.scatter(xi[0], xi[1], c='green', marker='^')
# plt.show()
# plt.axis('equal')


# 定义直线x + y = 0
# 圆圈是训练集数据，三角形是测试集数据，用颜色标记正负例
x = np.arange(-5, 5, 1)
plt.plot(x, -1 * x, c='red')

X = [[random.uniform(-5, 5), random.uniform(-5, 5)] for _ in range(20)]
y = [0] * len(X)
for i in range(len(X)):
    if X[i][0] + X[i][1] <= 0:
        y[i] = 1
        plt.scatter(X[i][0], X[i][1], c='blue', marker='o')
    else:
        y[i] = -1
        plt.scatter(X[i][0], X[i][1], c='green', marker='o')

K = getK(X, lambda x: x)
b = 0
alpha = [0] * len(X)
alpha,b = SMO(K, b, alpha, y)

for i in range(100):
    xi = [random.uniform(-5, 5), random.uniform(-5, 5)]
    pred = b
    for idx in range(len(alpha)):
        pred += alpha[idx] * y[idx] * kernelF(xi, X[idx], lambda x: x)
    if pred >= 0:
        plt.scatter(xi[0], xi[1], c='blue', marker='^')
    else:
        plt.scatter(xi[0], xi[1], c='green', marker='^')
plt.show()
plt.axis('equal')

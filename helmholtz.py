from math import cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4)

def f(x, s, l):
    return -(4 * s * pi**2 + l) * cos(2 * pi * x)

### Constants ###
lda = 1.0
sigma = 1.0

### Geometry ###
nDim = 1
leftCoor = 0.0
rightCoor = 1.0
length = rightCoor - leftCoor

### Boundary conditions ###
alpha = 1.0
beta = 0.0

### Integration ###
gaussOrder = 2
GaussPoints = [-1.0 / sqrt(3), 1.0 / sqrt(3)]
GaussWeights = [1.0, 1.0]

### Mesh ###
nElem = 5
nNodePerElem = 2
nNodeDof = [1, 1]
nDofInElem = sum(nNodeDof)
nNode = nElem + 1
nDof = nNode
deltaX = length / nElem

### Coordinate matrix ###
Coor = np.zeros((nNode, nDim))
for i in range(nNode):
    Coor[i, 0] = leftCoor + i * deltaX
# print Coor

### Connectivity matrix ###
Conn = np.zeros((nElem, nNodePerElem + 1), dtype=np.int)
for i in range(nElem):
    Conn[i, 0] = i
    Conn[i, 1] = i
    Conn[i, 2] = i + 1
# print Conn

### Global DOF ###
# GlobDof = np.zeros((nNode, 2), dtype=np.int)

### Assembly of M ###
M = np.zeros((nDof, nDof))
for e in range(nElem):
    nodesElem = Conn[e, 1:]
    # coorElem = Coor[nodesElem, :]
    Me = np.zeros((nNodePerElem, nNodePerElem))
    for i in range(gaussOrder):
        xi = GaussPoints[i]
        weight = GaussWeights[i]
        N = np.array([[(1.0 - xi) / 2.0], [(1.0 + xi) / 2.0]])
        Je = deltaX / 2.0
        Me = Me + N * N.T * Je * weight
    # print Me
    M[np.ix_(nodesElem, nodesElem)] = M[np.ix_(nodesElem, nodesElem)] + Me
# print M

### Assembly of L ###
L = np.zeros((nDof, nDof))
for e in range(nElem):
    nodesElem = Conn[e, 1:]
    # coorElem = Coor[nodesElem, :]
    Le = np.zeros((nNodePerElem, nNodePerElem))
    for i in range(gaussOrder):
        xi = GaussPoints[i]
        weight = GaussWeights[i]
        DerN = np.array([[-1.0 / 2.0], [1.0 / 2.0]])
        Je = deltaX / 2.0
        derXiDerX = 1.0 / Je
        Le = Le + DerN * DerN.T * derXiDerX * derXiDerX * Je * weight
    # print Le
    L[np.ix_(nodesElem, nodesElem)] = L[np.ix_(nodesElem, nodesElem)] + Le
# print L

### RHS terms ###
F = np.zeros((nDof, 1))
for e in range(nElem):
    nodesElem = Conn[e, 1:]
    coorElem = Coor[nodesElem, :]
    Fe = np.zeros((nNodePerElem, 1))
    for i in range(gaussOrder):
        xi = GaussPoints[i]
        weight = GaussWeights[i]
        N = np.array([[(1.0 - xi) / 2.0], [(1.0 + xi) / 2.0]])
        Je = deltaX / 2.0
        x = N[0] * coorElem[0] + N[1] * coorElem[1]
        Fe = Fe + N * f(x, sigma, lda) * Je * weight
    # print Fe
    F[np.ix_(nodesElem)] = F[np.ix_(nodesElem)] + Fe
F = -F
F[-1] = F[-1] + sigma * beta
# print F

### Partition matrices ###
DNodes = [0]
U = np.zeros((nDof, 1))
U[0] = alpha
K = L + lda * M
mask_D = np.array([(i in DNodes) for i in range(len(U))])
U_D = U[mask_D]
# print U_D
F_H = F[~mask_D]
# print F_H
K_DD = K[np.ix_(mask_D, mask_D)]
# print K_DD
K_HH = K[np.ix_(~mask_D, ~mask_D)]
# print K_HH
K_DH = K[np.ix_(mask_D, ~mask_D)]
# print K_DH

### Solve ###
RHS = F_H - np.dot(K_DH.T, U_D)
U_H = np.linalg.solve(K_HH, RHS)
# print U_H

### Reconstruct ###
U[mask_D] = U_D
U[~mask_D] = U_H
F_D = np.dot(K_DD, U_D) + np.dot(K_DH, U_H)
F[mask_D] = F_D
F[~mask_D] = F_H
# print U
# print F

### Error ###
Uex = np.cos(2 * pi * Coor)
# print Uex
err = 0.0
for i in range(nElem):
    err = err + (Uex[i] - U[i])**2 / (nNode)
err = sqrt(err)
print err

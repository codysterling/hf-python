#!/usr/bin/env python3

import sys
import numpy as np
import scipy as sp
from scipy.special import erf

######################
# Defining functions #
######################

def read_xyz(file):
  atoms = []
  coords = []

  with open(file) as f:
    natoms = int(f.readline().strip())
    f.readline()
    for line in f.readlines():
      words = line.split()
      atoms.append(words[0])
      coords.append([float(words[1]),float(words[2]),float(words[3])])
  return natoms, atoms, coords

def gProd(gA,gB):
  a, Ra = gA
  b, Rb = gB
  sqdiff = np.linalg.norm(Ra-Rb)**2
  p = a + b
  Rp = (a*Ra + b*Rb)/p
  N = (4*a*b/(np.pi**2))**0.75
  K = N*np.exp(-a*b/p*sqdiff)
  return p, sqdiff, K, Rp

def overlap(A,B):
  p, sqdiff, K, Rp = gProd(A,B)
  return (np.pi/p)**1.5*K

def kinetic(A,B):
  p, sqdiff, K, Rp = gProd(A,B)
  a, Ra = A
  b, Rb = B
  return (a*b/p)*(3-2*(a*b/p)*sqdiff)*overlap(A,B)

def F0(t):
  if t == 0:
    return 1
  else:
    return (0.5)*(np.pi/t)**0.5*erf(t**0.5)

def ne_pot(A,B,i):
  p, sqdiff, K, Rp = gProd(A,B)
  Rc = coords[i]
  Zc = charges[atoms[i]]
  return (-2*np.pi/p)*Zc*K*F0(p*(np.linalg.norm(Rp-Rc)**2))

def abcd(A,B,C,D):
  p, sqdiff_ab, K_ab, Rp = gProd(A,B)
  q, sqdiff_cd, K_cd, Rq = gProd(C,D)
  return 2*(np.pi**2.5)/(p*q*(p+q)**0.5)*K_ab*K_cd*F0(p*q/(p+q)*np.linalg.norm(Rp-Rq)**2)

def diff(P,Pt):
  x = 0
  for i in range(basis_size):
    for j in range(basis_size):
      x += ((Pt[i,j]-P[i,j])/basis_size)**2
  return x**0.5

######################

natoms, atoms, coords = read_xyz(sys.argv[1])

zetas = {'H': [1.24], 'He': [2.0925], 'Li': [2.69, 0.75], 'Be': [3.68, 1.10], 'B': [4.68, 1.45, 1.45, 1.45, 1.45], 'C': [5.67, 1.72, 1.72, 1.72, 1.72], 'N': [6.67, 1.95, 1.95, 1.95, 1.95], 'O': [7.66, 2.25, 2.25, 2.25, 2.25], 'F': [8.65, 2.55, 2.55, 2.55, 2.55]}

# STO-1G for H
#d_list = np.array([[1.0]])
#a_list = np.array([[0.270950]])

# STO-2G for H
#d_list = np.array([[0.678914, 0.430129]])
#a_list = np.array([[0.151623, 0.851819]])

# STO-3G
d_list = np.array([[0.444635, 0.535328, 0.154329], [0.700115, 0.399513, -0.0999672], [0.391957, 0.607684, 0.155916], [0.391957, 0.607684, 0.155916], [0.391957, 0.607684, 0.155916]]) # 1s, 2s, 2p
a_list = np.array([[0.109818, 0.405771, 2.22766], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203]]) # 1s, 2sp

basis_size = 0
for atom in atoms:
#  basis_size += len(np.unique(zetas[atom]))
  basis_size += len(zetas[atom])
print('Basis size: {}'.format(basis_size))

n_elec = 2
charges = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

# Making F matrices for overlap (S), kinetic energies (T), and potential energies (V), and multi-electron tensor (M)
S = np.zeros((basis_size,basis_size))
T = np.zeros((basis_size,basis_size))
V = np.zeros((basis_size,basis_size))
M = np.zeros((basis_size,basis_size,basis_size,basis_size))

for i,atom in enumerate(atoms):

  # Get atom charge and coordinates for centering the basis functions
  Za = charges[atom]
  Ra = np.array(coords[i])
  print('For atom {} {} we have Za = {} and Ra = {}'.format(i,atom,Za,Ra))

  # Iterating through quantum numbers on site A (1s, 2s, etc.)
  for qA in range(len(zetas[atom])):
    d_A = d_list[qA]
    zeta_Aq = zetas[atom][qA]
    a_A = a_list[qA]*zeta_Aq**2
    print('qA is {}, d_A = {}, zeta_Aq = {}, a_A = {}'.format(qA,d_A,zeta_Aq,a_A))

    for lA in range(len(d_list[0])): # Iterating over contracted Gaussians for A
      # Iterating through B atoms
      for j,atomB in enumerate(atoms):
        Zb = charges[atomB]
        Rb = np.array(coords[j])
        print('For atom {} {} we have Zb = {} and Rb = {}'.format(j,atomB,Zb,Rb))

        # Iterating through quantum numbers on site B (1s, 2s, etc.)
        for qB in range(len(zetas[atomB])):
          d_B = d_list[qB]
          zeta_Bq = zetas[atomB][qB]
          a_B = a_list[qB]*zeta_Bq**2
          print('d_B = {}, zeta_Bq = {}, a_B = {}'.format(d_B,zeta_Bq,a_B))

          for lB in range(len(d_list[0])): # Iterating over contracted Gaussians for B
            a = (i+1)*(qA+1)-1
            b = (j+1)*(qB+1)-1
            print(' ! i: {}, qA: {}, lA: {}, j: {}, qB: {}, lB: {}'.format(i,qA,lA,j,qB,lB))
            print(' ! a: {} b: {}'.format(a,b))

            S[a,b] += d_A[lA]*d_B[lB]*overlap((a_A[lA],Ra),(a_B[lB],Rb))
            T[a,b] += d_A[lA]*d_B[lB]*kinetic((a_A[lA],Ra),(a_B[lB],Rb))

            for n in range(natoms):
              V[a,b] += d_A[lA]*d_B[lB]*ne_pot((a_A[lA],Ra),(a_B[lB],Rb),n)

#            for k,atomC in enumerate(atoms): # Going through C and D to make the M tensor
#              Zc = charges[atomC]
#              Rc = np.array(coords[k])
#              print('For atom {} {} we have Zc = {} and Rc = {}'.format(k,atomC,Zc,Rc))

              # Iterating through quantum numbers on site C (1s, 2s, etc.)
#              for qC in range(len(zetas[atomC])):
#                d_C = d_list[qC]
#                zeta_Cq = zetas[atomC][qC]
#                a_C = a_list[qC]*zeta_Cq**2
#                print('d_C = {}, zeta_Cq = {}, a_C = {}'.format(d_C,zeta_Cq,a_C))

#                for lC in range(len(d_list[0])): # Iterating over contracted Gaussians for C

#                  for m,atomD in enumerate(atoms):
#                    Zd = charges[atomD]
#                    Rd = np.array(coords[m])
#                    print('For atom {} {} we have Zd = {} and Rd = {}'.format(m,atomD,Zd,Rd))

                    # Iterating through quantum numbers on site D (1s, 2s, etc.)
#                    for qD in range(len(zetas[atomD])):
#                      d_D = d_list[qD]
#                      zeta_Dq = zetas[atomD][qD]
#                      a_D = a_list[qD]*zeta_Dq**2
#                      print('d_D = {}, zeta_Dq = {}, a_D = {}'.format(d_D,zeta_Dq,a_D))

#                      for lD in range(len(d_list[0])): # Iterating over contracted Gaussians for D
#                        c = (k+1)*(qC+1)-1
#                        d = (m+1)*(qD+1)-1
#                        print(' ! k: {}, qC: {}, lC: {}, m: {}, qD: {}, lD: {}'.format(k,qC,lC,m,qD,lD))
#                        print(' ! c: {} d: {}'.format(c,d))
#                        M[a,b,c,d] += d_A[lA]*d_B[lB]*d_C[lC]*d_D[lD]*abcd((a_A[lA],Ra),(a_B[lB],Rb),(a_C[lC],Rc),(a_D[lD],Rd))

print('S:\n{}\n'.format(S))
print('T:\n{}\n'.format(T))
print('V:\n{}\n'.format(V))
print('M:\n{}\n'.format(M))

Hcore = T + V

print('Hcore:\n{}\n'.format(Hcore))

# Orthogonalizing basis
evalS, U = np.linalg.eig(S)
#evalS, U = sp.linalg.schur(S)
#U[:,1] *= -1
print('evalS:\n{}\n'.format(evalS))
print('U:\n{}\n'.format(U))
print('U.T:\n{}\n'.format(U.T))

diagS = np.dot(U.T,np.dot(S,U))
print('diagS:\n{}\n'.format(diagS))
diagS_minushalf = np.diag(np.diag(diagS)**-0.5)
print('diagS_minushalf:\n{}\n'.format(diagS_minushalf))
X = np.dot(U,np.dot(diagS_minushalf,U.T))
X = np.dot(U,diagS_minushalf)
print('X:\n{}\n'.format(X))

P = np.zeros((basis_size,basis_size))
Pt = np.zeros((basis_size,basis_size))
P_list = []
E_list = []

threshold = 100
while threshold > 99:
  print('Iteration: {}'.format(len(P_list)))
  G = np.zeros((basis_size,basis_size))
  for i in range(basis_size):
    for j in range(basis_size):
      for x in range(basis_size):
        for y in range(basis_size):
          G[i,j] += P[x,y]*(M[i,j,y,x]-0.5*M[i,x,y,j])
  print('G:\n{}\n'.format(G))
  F = Hcore + G
  print('F:\n{}\n'.format(F))

  # Calculate energy
  E0 = 0
  for u in range(basis_size):
    for v in range(basis_size):
      E0 += 0.5*P[v,u]*(Hcore[u,v] + F[u,v])
  E_list.append(E0)
  print('Energy: {}'.format(E0))

  Fprime = np.dot(X.T,np.dot(F,X))
  print('Fprime:\n{}\n'.format(Fprime))
  evalFprime, Cprime = np.linalg.eig(Fprime)
  print('evalFprime:\n{}\n'.format(evalFprime))
  print('Cprime:\n{}\n'.format(Cprime))

  ## Necessary?
  evalFprime = evalFprime[evalFprime.argsort()]
  Cprime = Cprime[:,evalFprime.argsort()]
  print('evalFprime again:\n{}\n'.format(evalFprime))
  print('Cprime again:\n{}\n'.format(Cprime))

  C = np.dot(X,Cprime)
  print('C:\n{}\n'.format(C))

  # Iterating P
  for i in range(basis_size):
    for j in range(basis_size):
      for a in range(int(n_elec/2)):
        print('Cia = C{}{} = {}'.format(i,a,C[i,a]))
        print('Cja = C{}{} = {}'.format(j,a,C[j,a]))
        P[i,j] = 2*C[i,a]*C[j,a]
  print('P:\n{}\n'.format(P))

  P_list.append(P)
  threshold = diff(P,Pt)
  print('Threshold is {}'.format(threshold))
  Pt = P.copy()

print("Converged in {} iterations".format(len(P_list)))
print("Orbital energies are {}".format(evalFprime))
print("Orbital matrix is {}".format(C))
print("Density bond order matrix is {}".format(P))
print("Energy is {}".format(E_list[-1]))
print("Energy list is {}".format(E_list))

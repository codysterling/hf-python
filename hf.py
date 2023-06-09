#!/usr/bin/env python3

import sys
import numpy as np
import math

######################
# Defining functions #
######################

def readXYZ(file):
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
		return (0.5)*(np.pi/t)**0.5*math.erf(t**0.5)

def nePotential(A,B,i):
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

def nuclearRepulsion():
	nuc_rep = 0
	for i,A in enumerate(atoms):
		print("i is {} {}".format(i,A))
		for j,B in enumerate(atoms):
			print("j is {} {}".format(j,B))
			if i == j:
				continue
			charge_A = charges[A]
			charge_B = charges[B]
			Ra = np.array(coords[i])
			Rb = np.array(coords[j])
			R = np.linalg.norm(Ra-Rb)
			print(Ra-Rb,R)
			nuc_rep += (charge_A*charge_B)/R
	return 0.5*nuc_rep

natoms, atoms, coords = readXYZ(sys.argv[1])
#natoms, atoms, coords = readXYZ("/home/cody/Documents/Gitlab/hf-python/xyz-files/heh.xyz")

zetas = {'H': [1.24], 'He': [2.0925], 'Li': [2.69, 0.75], 'Be': [3.68, 1.10], 'B': [4.68, 1.45, 1.45, 1.45, 1.45], 'C': [5.67, 1.72, 1.72, 1.72, 1.72], 'N': [6.67, 1.95, 1.95, 1.95, 1.95], 'O': [7.66, 2.25, 2.25, 2.25, 2.25], 'F': [8.65, 2.55, 2.55, 2.55, 2.55]}

# matching ORCA
#zetas = {'H': [1.24], 'He': [2.0925], 'Li': [2.69, 0.8, 0.8], 'Be': [3.68, 1.10], 'B': [4.68, 1.45, 1.45, 1.45, 1.45], 'C': [5.67, 1.72, 1.72, 1.72, 1.72], 'N': [6.67, 1.95, 1.95, 1.95, 1.95], 'O': [7.66, 2.25, 2.25, 2.25, 2.25], 'F': [8.65, 2.55, 2.55, 2.55, 2.55]}

basis_size = 0
for atom in atoms:
	basis_size += len(zetas[atom])
print('Basis size: {}'.format(basis_size))

# STO-1G for H
#d_list = np.array([[1.0]])
#a_list = np.array([[0.270950]])

# STO-2G for H
#d_list = np.array([[0.678914, 0.430129]])
#a_list = np.array([[0.151623, 0.851819]])

# STO-3G
d_list = np.array([[0.444635, 0.535328, 0.154329], [0.700115, 0.399513, -0.0999672], [0.391957, 0.607684, 0.155916], [0.391957, 0.607684, 0.155916], [0.391957, 0.607684, 0.155916]]) # 1s, 2s, 2p
a_list = np.array([[0.109818, 0.405771, 2.22766], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203], [0.0751386, 0.231031, 0.994203]]) # 1s, 2sp

# Atom charges
charges = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

n_elec = 0
charge = 0
for atom in atoms:
	n_elec += charges[atom]
n_elec -= charge
print('nelec: {}'.format(n_elec))

# Making F matrices for overlap (S), kinetic energies (T), and potential energies (V), and multi-electron tensor (M)
S = np.zeros((basis_size,basis_size))
T = np.zeros((basis_size,basis_size))
V = np.zeros((basis_size,basis_size))
M = np.zeros((basis_size,basis_size,basis_size,basis_size))

a = 0
for i,atomA in enumerate(atoms):
	# Get atom charge and coordinates for centering the basis functions
	Za = charges[atomA]
	Ra = np.array(coords[i])
	print('For atom {} {} we have Za = {} and Ra = {}'.format(i,atomA,Za,Ra))

	# Iterating through quantum numbers on site A (1s, 2s, etc.)
	for qA,zeta_qA in enumerate(zetas[atomA]):
		b = 0
		d_qA = d_list[qA]
		a_qA = a_list[qA]*zeta_qA**2
		print('qA is {}, d_A = {}, zeta_Aq = {}, a_A = {}'.format(qA,d_qA,zeta_qA,a_qA))

		# Iterating through B atoms
		for j,atomB in enumerate(atoms):
			
			Zb = charges[atomB]
			Rb = np.array(coords[j])
			print('For atom {} {} we have Zb = {} and Rb = {}'.format(j,atomB,Zb,Rb))

			# Iterating through quantum numbers on site B (1s, 2s, etc.)
			for qB,zeta_qB in enumerate(zetas[atomB]):
				c = 0
				d_qB = d_list[qB]
				a_qB = a_list[qB]*zeta_qB**2
				print('qB is {}, d_B = {}, zeta_Bq = {}, a_B = {}'.format(qB,d_qB,zeta_qB,a_qB))

				for lA in range(len(d_qA)): # Iterating over contracted Gaussians for A
					for lB in range(len(d_qB)): # Iterating over contracted Gaussians for B
						# a = (i+1)*(qA+1)-1
						# b = (j+1)*(qB+1)-1
						print(' ! i: {}, qA: {}, lA: {}, j: {}, qB: {}, lB: {}'.format(i,qA,lA,j,qB,lB))
						print(' ! a: {} b: {}'.format(a,b))

						S[a,b] += d_qA[lA]*d_qB[lB]*overlap((a_qA[lA],Ra),(a_qB[lB],Rb))
						T[a,b] += d_qA[lA]*d_qB[lB]*kinetic((a_qA[lA],Ra),(a_qB[lB],Rb))

						for n in range(natoms):
							V[a,b] += d_qA[lA]*d_qB[lB]*nePotential((a_qA[lA],Ra),(a_qB[lB],Rb),n)

				# Going through C and D to make the M tensor
				for k,atomC in enumerate(atoms):
					Zc = charges[atomC]
					Rc = np.array(coords[k])
					print('For atom {} {} we have Zc = {} and Rc = {}'.format(k,atomC,Zc,Rc))

					# Iterating through quantum numbers on site C (1s, 2s, etc.)
					for qC,zeta_qC in enumerate(zetas[atomC]):
						d = 0
						d_qC = d_list[qC]
						a_qC = a_list[qC]*zeta_qC**2
						print('d_C = {}, zeta_Cq = {}, a_C = {}'.format(d_qC,zeta_qC,a_qC))

						for m,atomD in enumerate(atoms):
							Zd = charges[atomD]
							Rd = np.array(coords[m])
							print('For atom {} {} we have Zd = {} and Rd = {}'.format(m,atomD,Zd,Rd))

							# Iterating through quantum numbers on site D (1s, 2s, etc.)
							for qD,zeta_qD in enumerate(zetas[atomD]):
								d_qD = d_list[qD]
								a_qD = a_list[qD]*zeta_qD**2
								print('d_D = {}, zeta_Dq = {}, a_D = {}'.format(d_qD,zeta_qD,a_qD))

								for lA in range(len(d_qA)): # Iterating over contracted Gaussians for A
									for lB in range(len(d_qB)): # Iterating over contracted Gaussians for B
										for lC in range(len(d_qC)): # Iterating over contracted Gaussians for C
											for lD in range(len(d_qD)): # Iterating over contracted Gaussians for D
												# c = (k+1)*(qC+1)-1
												# d = (m+1)*(qD+1)-1
												print(' ! k: {}, qC: {}, lC: {}, m: {}, qD: {}, lD: {}'.format(k,qC,lC,m,qD,lD))
												print(' ! c: {} d: {}'.format(c,d))
												M[a,b,c,d] += d_qA[lA]*d_qB[lB]*d_qC[lC]*d_qD[lD]*abcd((a_qA[lA],Ra),(a_qB[lB],Rb),(a_qC[lC],Rc),(a_qD[lD],Rd))

								
								d += 1
								print('hered {}'.format(d))
						
						c += 1
						print('herec {}'.format(c))
				
				b += 1
				print('hereb {}'.format(b))
		
		a += 1
		print('herea {}'.format(a))


print('S:\n{}\n'.format(S))
print('Ssum: {}'.format(np.sum(S)))
print('T:\n{}\n'.format(T))
print('Tsum: {}'.format(np.sum(T)))
print('V:\n{}\n'.format(V))
print('Vsum: {}'.format(np.sum(V)))
print('M:\n{}\n'.format(M))
print('Msum: {}'.format(np.sum(M)))

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
while threshold > 1e-6:
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
				print('Cia = C{}{} = {:0.5f}'.format(i,a,C[i,a]))
				print('Cja = C{}{} = {:0.5f}'.format(j,a,C[j,a]))
				P[i,j] = 2*C[i,a]*C[j,a]
	print('P:\n{}\n'.format(P))

	P_list.append(P)
	threshold = diff(P,Pt)
	print('Threshold is {:0.5f}'.format(threshold))
	Pt = P.copy()

	# Calculate energy
	E0 = 0
	for u in range(basis_size):
		for v in range(basis_size):
			print("u, v: {}, {}".format(u,v))
			E0 += 0.5*P[v,u]*(Hcore[u,v] + F[u,v])
	E_list.append(E0)
	print('Energy: {:0.5f}'.format(E0))

print("Converged in {} iterations".format(len(P_list)))
print("Orbital energies are {}".format(evalFprime))
print("Orbital matrix is\n  {}".format(C))
print("Density bond order matrix is\n  {}".format(P))
print("Energy is {}".format(E_list[-1]))
print("Energy list is {}".format(E_list))
print("Nuclear repulsion is {}".format(nuclearRepulsion()))
print("Total energy is {}".format(E_list[-1]+nuclearRepulsion()))

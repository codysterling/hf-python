import numpy as np
import scipy as sp
from scipy.special import erf

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

def nuc_repul():
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
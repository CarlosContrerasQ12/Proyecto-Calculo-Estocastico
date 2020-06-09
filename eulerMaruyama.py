import numpy as np
import matplotlib.pyplot as plt
from numba import jit
T=1.0
N=10**1
dt=T/N
movimientos=10000
tiem=np.arange(0,T,dt)

mu=2.0
sigma=1.0
y0=1.0



@jit
def solucionarEuler(dMBa):
	solAprox=np.zeros(N)
	solAprox[0]=y0
	for i in range(N-1):
		solAprox[i+1]=solAprox[i]+mu*solAprox[i]*dt+sigma*solAprox[i]*(dMBa[i])
	return solAprox

@jit	
def solucionarMilstein(dMBa):
	solAprox=np.zeros(N+1)
	solAprox[0]=y0
	for i in range(N):
		solAprox[i+1]=solAprox[i]+(mu*solAprox[i]*dt)+(sigma*solAprox[i]*(dMBa[i]))+0.5*(sigma**2)*solAprox[i]*((dMBa[i]**2)-dt)
	return solAprox[1:]
	
@jit
def errores():
	solsum=np.zeros(N)
	soleul=np.zeros(N)
	solmils=np.zeros(N)
	errMil=np.zeros(N)
	errEul=np.zeros(N)
	for j in range(movimientos):
		np.random.seed(j)
		dMB=np.sqrt(dt)*np.random.randn(N)
		MB=np.cumsum(dMB)
		solExc=y0*np.exp((mu-0.5*sigma*sigma)*tiem+sigma*MB)
		
		solAprox=solucionarMilstein(dMB)
		errMil+=np.absolute(solAprox-solExc)
		
		solAprox2=solucionarEuler(dMB)
		errEul+=np.absolute(solAprox2-solExc)
		
		solsum+=solExc
		soleul+=solAprox2
		solmils+=solAprox

		
	errorMielFuerte=np.max(errMil/movimientos)
	errorEulerFuerte=np.max(errEul/movimientos)
	
	errorMielDebil=np.max(np.absolute(solsum-solmils)/movimientos)
	errorEulerDebil=np.max(np.absolute(solsum-soleul)/movimientos)
	return errorMielFuerte,errorEulerFuerte,errorMielDebil,errorEulerDebil
	
	
mf,ef,md,ed=errores()	
arch=open("datos.txt",'a')
arch.write(str(dt)+','+str(mf)+','+str(ef)+','+str(md)+','+str(ed)+'\n')
arch.close()



print(errores())


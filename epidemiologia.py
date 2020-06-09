import numpy as np
import matplotlib.pyplot as plt
from numba import jit
T=100.0
N=20000
dt=T/N
movimientos=10000
tiem=np.arange(0,T,dt)



@jit
def solucionar():
	alpha=0.04
	gamma=0.01
	promS=np.zeros(N)
	promI=np.zeros(N)
	for j in range(movimientos):
		np.random.seed(j)
		dMB=np.sqrt(dt)*np.random.randn(N)
		MB=np.cumsum(dMB)
		S=np.zeros(N)
		S[0]=950
		I=np.zeros(N)
		I[0]=50
		for i in range(1,N):
			m12=alpha*(I[i-1]/(I[i-1]+S[i-1]))
			m21=gamma
			S[i]=S[i-1]+(m21*I[i-1]-m12*S[i-1])*dt+np.sqrt(m12*S[i-1]+m21*I[i-1])*dMB[i-1]
			I[i]=I[i-1]+(m12*S[i-1]-m21*I[i-1])*dt-np.sqrt(m12*S[i-1]+m21*I[i-1])*dMB[i-1]
		promS+=S
		promI+=I
	promS=promS/movimientos
	promI=promI/movimientos
	print(promS[-1])
	print(promI[-1])
	plt.plot(tiem,promS,label='S(t)')
	plt.plot(tiem,promI,label='I(t)')
	
	np.random.seed(289)
	dMB=np.sqrt(dt)*np.random.randn(N)
	MB=np.cumsum(dMB)
	S=np.zeros(N)
	S[0]=950
	I=np.zeros(N)
	I[0]=50
	for i in range(1,N):
		m12=alpha*(I[i-1]/(I[i-1]+S[i-1]))
		m21=gamma
		S[i]=S[i-1]+(m21*I[i-1]-m12*S[i-1])*dt+np.sqrt(m12*S[i-1]+m21*I[i-1])*dMB[i-1]
		I[i]=I[i-1]+(m12*S[i-1]-m21*I[i-1])*dt-np.sqrt(m12*S[i-1]+m21*I[i-1])*dMB[i-1]
	plt.plot(tiem,S,ls='--')
	plt.plot(tiem,I,ls='--')
	plt.grid()
	plt.legend()
	plt.show()

solucionar()
			
			

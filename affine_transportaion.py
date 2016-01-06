import os
import numpy	as	np
import numpy.linalg	as	la

file	=	open('/Users/yuanruiliu/Documents/python/data.txt')
data	=	np.genfromtxt(file,	delimiter=",")
file.close()
print "Data:\n",data
n=len(data[:,0])
arrayM = np.array([data[0,0], data[0,1],1,0,0,0])
arrayM = np.vstack((arrayM, np.array([0,0,0,data[0,0], data[0,1],1])))
arrayB	=	np.array([[data[0,2]],	[data[0,3]]])
for	i in	range(1,n):
    arrayM = np.vstack((arrayM, np.array([data[i,0], data[i,1],1,0,0,0])))
    arrayM = np.vstack((arrayM, np.array([0,0,0,data[i,0], data[i,1],1])))
    arrayB = np.vstack((arrayB,data[i,2]))
    arrayB = np.vstack((arrayB,data[i,3]))
M = np.matrix(arrayM)
b = np.matrix(arrayB)
print "Matrix M:\n",M
print "Matrix b:\n",b
a,	e,	r,	s =	la.lstsq(M,	b)
print "Matrix a:\n",a
print "sum-squared error calculated by la.norm(M*a-b)^2:\n",la.norm(M*a-b)*la.norm(M*a-b)
print "residue(sum-squared error) computed by la.lstsq:\n",e
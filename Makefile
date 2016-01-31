# ---- Link --------------------------- 
_ising_1D.so:  ising_1D.o
	gcc -shared ising_1D.o -o _ising_1D.so  
# ---- gcc C compile ------------------
ising_1D.o:  ising_1D.c
	gcc  -c ising_1D.c -I/usr/include/python2.6 -I/usr/lib/python2.6/dist-packages/numpy/core/include/numpy


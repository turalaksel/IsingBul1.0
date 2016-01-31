# ---- Link --------------------------- 
_ising_1D.so:  ising_1D.o ising_1D.mak
	gcc -bundle -flat_namespace -undefined suppress -o _ising_1D.so ising_1D.o  
# ---- gcc C compile ------------------
ising_1D.o:  ising_1D.c ising_1D.mak
	gcc  -c ising_1D.c -I/Library/Frameworks/Python.framework/Versions/6.1/include/python2.6/ -I/Library/Frameworks/Python.framework/Versions/6.1/lib/python2.6/site-packages/numpy/core/include/numpy/


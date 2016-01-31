/***************************************************************************
 *   Copyright (C) 2007 by Tural Aksel   *
 *   tural@jhu.edu   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#include <Python.h>
#include <arrayobject.h>
#include <math.h>

//Put the function headers

static PyObject *ising_1D_simulate(PyObject *self,PyObject *args);
static PyObject *ising_1D_state(PyObject *self,PyObject *args);

static PyMethodDef _ising_1D_methods[] = {
    {"ising_1D_simulate", ising_1D_simulate, METH_VARARGS},
    {"ising_1D_state"   , ising_1D_state   , METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void init_ising_1D()  {
    (void) Py_InitModule("_ising_1D", _ising_1D_methods);
    import_array();  // Must be present for NumPy.  Called first after above line.
}



static PyObject *ising_1D_simulate(PyObject *self,PyObject *args)
{
   double kT;
   PyArrayObject *den_NumpyArray;
   PyArrayObject *G_intrn_NumpyArray;
   PyArrayObject *G_inter_NumpyArray;
   PyArrayObject *m_intrn_NumpyArray;
   PyArrayObject *m_inter_NumpyArray;
   PyArrayObject *fracFolded_NumpyArray;
   
   int i,nRepeats,nDen,dims[2];
   double *den;
   double *G_intrn;
   double *G_inter;
   double *m_intrn;
   double *m_inter;
   double **fracFolded;
   
   //Read the parameters passed from python
   if (!PyArg_ParseTuple(args, "dO!O!O!O!O!",&kT,&PyArray_Type,&den_NumpyArray,&PyArray_Type,&G_intrn_NumpyArray,&PyArray_Type,&G_inter_NumpyArray,&PyArray_Type,&m_intrn_NumpyArray,&PyArray_Type,&m_inter_NumpyArray))
   {
     return NULL;
   }
   
   //Get the ctype values of each array/variable
   nRepeats = G_intrn_NumpyArray->dimensions[0];
   nDen     = den_NumpyArray->dimensions[0];
   
   den      = (double *) den_NumpyArray->data; 
   G_intrn  = (double *) G_intrn_NumpyArray->data;
   G_inter  = (double *) G_inter_NumpyArray->data;
   m_intrn  = (double *) m_intrn_NumpyArray->data;
   m_inter  = (double *) m_inter_NumpyArray->data;
   
   //Prepare fracFolded
   dims[0] = nRepeats;
   dims[1] = nDen;
   fracFolded_NumpyArray = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
   fracFolded    = (double **) calloc(dims[0],sizeof(double *));
   for(i = 0; i<dims[0]; i++)
   {
     fracFolded[i] = (double *) fracFolded_NumpyArray->data + i*dims[1];
   }
   //Initialize the 2D arrays Z_forward and Z_backward
   double Z_total,Z_i,inter_coef,intrn_coef;
   int    d,forward,reverse;
  
   double **Z_forward;
   double **Z_reverse;
   
   Z_forward = (double **) calloc (nRepeats+2,sizeof(double *));
   for(i = 0; i<nRepeats+2; i++)
   {
     Z_forward[i] = (double *) calloc (2,sizeof(double));
   }
  
   Z_reverse = (double **) calloc (nRepeats+2,sizeof(double *));
   for(i = 0; i<nRepeats+2; i++)
   {
     Z_reverse[i] = (double *) calloc (2,sizeof(double));
   }
  
   //We will compute the partition function for each fraction	1->i : Z_forward
   //								n->i : Z_reverse
  
   for( d = 0; d < nDen;d++)
   {
     // Initialization of the position specific partition functions
	 // Z_forward will be 1X2 row vector
	 // Z_reverse will be 2X1 column vector
	
	Z_forward[0][0] = 0;
	Z_forward[0][1] = 1;
	Z_reverse[nRepeats+1][0] = 1;
	Z_reverse[nRepeats+1][1] = 1; 
	
	forward = 1;
	reverse = nRepeats;
	for( i = 0; i < nRepeats; i++)
	{
	  
	  intrn_coef  =  exp((G_intrn[i]-m_intrn[i]*den[d])/ kT);
	  inter_coef  =  exp((G_inter[i]-m_inter[i]*den[d])/ kT);
	  Z_forward[forward][0]  = intrn_coef*inter_coef*Z_forward[forward-1][0]+ intrn_coef*Z_forward[forward-1][1];
	  Z_forward[forward][1]  = Z_forward[forward-1][0]+ Z_forward[forward-1][1];
	  intrn_coef  =  exp((G_intrn[nRepeats-1-i]-m_intrn[nRepeats-1-i]*den[d])/ kT);
	  inter_coef  =  exp((G_inter[nRepeats-1-i]-m_inter[nRepeats-1-i]*den[d])/ kT);
	  Z_reverse[reverse][0] = inter_coef*intrn_coef*Z_reverse[reverse+1][0] + Z_reverse[reverse+1][1];
	  Z_reverse[reverse][1] = intrn_coef*Z_reverse[reverse+1][0] + Z_reverse[reverse+1][1];
	  forward++;
	  reverse--;
	}
	//Find the total partition function
	Z_total = Z_reverse[1][1];
	//Now calculate the probability  of seeing each repeat folded
	for (i = 0; i< nRepeats; i++)
	{
	  intrn_coef  =  exp((G_intrn[i]-m_intrn[i]*den[d])/ kT);
	  inter_coef  =  exp((G_inter[i]-m_inter[i]*den[d])/ kT);
	  Z_i  = Z_reverse[i+2][0]*(Z_forward[i][0]*intrn_coef*inter_coef + Z_forward[i][1]*intrn_coef);
	  fracFolded[i][d] = Z_i/Z_total;
	}
	  
  }
  //Free allocated memories
  for (i = 0; i < nRepeats+2; i++)
  {	
	free(Z_forward[i]);
	free(Z_reverse[i]);
  }
  free(Z_forward);
  free(Z_reverse);
  free(fracFolded);
  return PyArray_Return(fracFolded_NumpyArray);
}

static PyObject *ising_1D_state(PyObject *self,PyObject *args)
{
   double kT;
   PyArrayObject *den_NumpyArray;
   PyArrayObject *state_NumpyArray;
   PyArrayObject *G_intrn_NumpyArray;
   PyArrayObject *G_inter_NumpyArray;
   PyArrayObject *m_intrn_NumpyArray;
   PyArrayObject *m_inter_NumpyArray;
   PyArrayObject *Z_total_NumpyArray;
   
   
   //State keeps the information about the state of each spin/repeat
   //1: Repeat can be folded/unfolded
   //0: Repeat can be unfolded only
   
   int i,nRepeats,nDen,dims[2];
   double *state;
   double *den;
   double *G_intrn;
   double *G_inter;
   double *m_intrn;
   double *m_inter;
   double *Z_total;
   double **fracFolded;
   double temp;
   
   
   //Read the parameters passed from python
   if (!PyArg_ParseTuple(args, "dO!O!O!O!O!O!",&kT,&PyArray_Type,&state_NumpyArray,&PyArray_Type,&den_NumpyArray,&PyArray_Type,&G_intrn_NumpyArray,&PyArray_Type,&G_inter_NumpyArray,&PyArray_Type,&m_intrn_NumpyArray,&PyArray_Type,&m_inter_NumpyArray))
   {
     return NULL;
   }
   
   //Get the ctype values of each array/variable
   nRepeats = G_intrn_NumpyArray->dimensions[0];
   nDen     = den_NumpyArray->dimensions[0];
   state    = (double *) state_NumpyArray->data;
   den      = (double *) den_NumpyArray->data; 
   G_intrn  = (double *) G_intrn_NumpyArray->data;
   G_inter  = (double *) G_inter_NumpyArray->data;
   m_intrn  = (double *) m_intrn_NumpyArray->data;
   m_inter  = (double *) m_inter_NumpyArray->data;
   
   //Prepare Z_total
   dims[0] = 1;
   dims[1] = nDen;
   Z_total_NumpyArray = (PyArrayObject *) PyArray_SimpleNew(2,dims,NPY_DOUBLE);
   Z_total = (double *) Z_total_NumpyArray->data;
   
   //Initialize the arrays: Z_forward 
   double inter_coef,intrn_coef;
   int    d;
  
   double *Z_forward;
   Z_forward = (double *) calloc (2,sizeof(double));
  
   //We will compute the partition function for each fraction	1->i : Z_forward
   //								
  
   for( d = 0; d < nDen;d++)
   {
     // Initialization of the position specific partition functions
	 // Z_forward will be 1X2 row vector
	 // Z_reverse will be 2X1 column vector
	
	Z_forward[0] = 0;
	Z_forward[1] = 1;
	
	for( i = 0; i < nRepeats; i++)
	{
	  intrn_coef  =  exp((G_intrn[i]-m_intrn[i]*den[d])/ kT);
	  inter_coef  =  exp((G_inter[i]-m_inter[i]*den[d])/ kT);
	  temp        = Z_forward[0];
	  if(state[i] == 1)	//Repeat is folded
	  {
	    Z_forward[0]  = intrn_coef*inter_coef*Z_forward[0]+ intrn_coef*Z_forward[1];
	    Z_forward[1]  = 0;
	  }
	  else if(state[i] == -1)//Repeat is unfolded
	  {
	    Z_forward[0]  = 0;
	    Z_forward[1]  = temp + Z_forward[1];
	  }
	  else if(state[i] == 0)//Repeat can be both
	  {
	    Z_forward[0]  = intrn_coef*inter_coef*Z_forward[0]+ intrn_coef*Z_forward[1];
	    Z_forward[1]  = temp + Z_forward[1];
	  }
	}
	//Find the total partition function
	Z_total[d] = Z_forward[0]+Z_forward[1];
    }
  //Free allocated memories
  free(Z_forward);
  return PyArray_Return(Z_total_NumpyArray);
}

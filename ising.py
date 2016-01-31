#!/usr/bin/env python

#A collection of codes for reading, processing and fitting data
#that we collect using AVIV CD, Applied Photophysics SF and AVIV ATF instruments

#Tural Aksel
#02/20/09

import _ising_1D as ising
import numpy as nm
from numpy.linalg import svd
import matplotlib.pyplot as py
import matplotlib.cm as cm
from scipy import optimize
import re
import time
from ctypes import *

#Plotting parameters

params = {'axes.labelsize' : 20,
          'legend.fontsize': 20,
          'axes.titlesize' : 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'figure.figsize' : (5.5,5.5),
          'figure.dpi'     : 100,
          'lines.linewidth':3
          }
py.rcParams.update(params)

yTick = {'major.pad': 10,'major.size': 8,'minor.size': 6}
xTick = {'major.pad': 10,'major.size': 8,'minor.size': 6}

py.rc('xtick', **xTick)  # pass in the font dict as kwargs
py.rc('ytick', **yTick)  # pass in the font dict as kwargs

#Several constants commonly used in thermodynamics
''' boltzmann constant in kcal/(K*mol)'''
_K_ = 1.9858775E-3


#Some global utility functions
def normalize_min_max(data):
    '''
    Normalizes the data using min/max
    '''
    return (data - nm.min(data))/(nm.max(data) - nm.min(data))

def is_number(string):
    '''
    Checks if a string is in float format or not
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False

def matrix_rank(A,tol=1e-8):
    '''
    Returns the rank of a matrix
    '''
    s = svd(A,compute_uv=0)
    return sum( nm.where( s>tol, 1, 0 ) )
def c_double2pointer(value):
    '''
    Returns a pointer to a c_double
    '''
    return pointer(c_double(value))

def c_int2pointer(value):
    '''
    Returns a pointer to a c_int
    '''
    return pointer(c_int(value))
    
def pointer2value(pointer):
    '''
    Returns the value that pointer points to
    '''
    return pointer.contents.value

class IsingDenData:
    '''
    Equilibrium titration data container for ising analysis
    '''
    def _init_(self):
        self.num_repeats    = 0
        self.protein        = ''
        self.repeats        = []
        self.inters         = []
        self.denaturant     = nm.array([])
        self.exp_signal     = nm.array([])
        self.est_signal     = nm.array([])
        self.frac_folded    = nm.array([])
        self.corr_matrix    = nm.array([])
        self.ave_folded     = nm.array([])
        self.residuals      = nm.array([])
        self.G_intrin       = []
        self.m_intrin       = []
        self.G_inter        = []
        self.m_inter        = []
        self.weight         = 1
        self.signal_type    = 'CD'
        self.temp           = c_double2pointer(298.15)
        self.kT             = c_double2pointer(_K_*298.15)
        self.Mf             = c_double2pointer(0.0)
        self.Af             = c_double2pointer(0.0)
        self.Mu             = c_double2pointer(0.0)
        self.Au             = c_double2pointer(0.0)
    
    def copy(self,other):
        '''
        Deep copy the contents of the other to self
        '''
        self.num_repeats = other.num_repeats
        self.protein     = other.protein
        self.repeats     = other.repeats
        self.inters      = other.inters
        self.denaturant  = other.denaturant
        self.exp_signal  = other.exp_signal  + nm.zeros(other.exp_signal.shape)
        self.est_signal  = other.est_signal  + nm.zeros(other.est_signal.shape)
        self.frac_folded = other.frac_folded + nm.zeros(other.frac_folded.shape)
        self.ave_folded  = other.ave_folded  + nm.zeros(other.ave_folded.shape)
        self.residuals   = other.residuals   + nm.zeros(other.residuals.shape)
        self.G_intrin    = other.G_intrin
        self.m_intrin    = other.m_intrin
        self.G_inter     = other.G_inter
        self.m_inter     = other.m_inter
        self.weight      = other.weight
        self.signal_type = other.signal_type
        self.temp        = other.temp
        self.kT          = other.kT
        self.Mf          = other.Mf
        self.Af          = other.Af
        self.Mu          = other.Mu
        self.Au          = other.Au
        
    def read_data(self,data_file,signal_type='CD'):
        '''
        Read titration data file
        '''
        new_titration = Titration()
        new_titration.read_data(data_file)
        
        self.denaturant  = new_titration.titrant
        self.exp_signal  = new_titration.cd_signal
        self.signal_type = signal_type
        if signal_type == 'FL':
            self.exp_signal  = new_titration.fl_signal
    
    def decompose_components(self):
        '''
        Identify the repeat and interface names of the repeat protein
        '''
        self.repeats= []
        self.inters = []
        repeat_pre = ''
        for repeat in self.protein:
            inter = repeat_pre+repeat
            self.repeats.append(repeat)
            if not repeat_pre == '':
                self.inters.append(inter)
            repeat_pre = repeat
    
    def simulate(self):
        '''
        Simulate the titration curve given the thermodynamic parameters
        '''
        G_intrin = nm.array([param.contents.value for param in self.G_intrin])
        G_inter  = nm.array([param.contents.value for param in self.G_inter])
        m_intrin = nm.array([param.contents.value for param in self.m_intrin])
        m_inter  = nm.array([param.contents.value for param in self.m_inter])
        
        kT = self.kT.contents.value
        
        Mf = self.Mf.contents.value
        Af = self.Af.contents.value
        Mu = self.Mu.contents.value
        Au = self.Au.contents.value
        
        self.frac_folded  = ising.ising_1D_simulate(kT,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)
        self.ave_folded   = nm.sum(self.frac_folded,axis=0)/self.num_repeats        
        self.est_signal   = (Mf*self.denaturant+Af)*self.ave_folded + (Mu*self.denaturant+Au)*(1-self.ave_folded)
    
    def correlation_matrix(self,denaturant):
        
        
        #Initialize the correlation matrix
        self.corr_matrix = nm.zeros((self.num_repeats,self.num_repeats))
        
        self.denaturant = denaturant
        
        G_intrin = nm.array([param.contents.value for param in self.G_intrin])
        G_inter  = nm.array([param.contents.value for param in self.G_inter])
        m_intrin = nm.array([param.contents.value for param in self.m_intrin])
        m_inter  = nm.array([param.contents.value for param in self.m_inter])
        
        kT = self.kT.contents.value
        
        Mf = self.Mf.contents.value
        Af = self.Af.contents.value
        Mu = self.Mu.contents.value
        Au = self.Au.contents.value
        
        state   = nm.zeros(self.num_repeats)
        
        Z_total = ising.ising_1D_state(kT,state,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)    
        for repeat1 in range(self.num_repeats):
            state = nm.zeros(self.num_repeats)
            state[repeat1] = 1
            Z_1_folded   = ising.ising_1D_state(kT,state,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)
            Z_1_unfolded = Z_total - Z_1_folded
            
            #Calculate self correlation matrix
            self.corr_matrix[repeat1,repeat1] = (Z_1_folded+Z_1_unfolded)/Z_total - (Z_1_folded-Z_1_unfolded)**2/(Z_total)**2
            
            for repeat2 in range(repeat1+1,self.num_repeats):
                state = nm.zeros(self.num_repeats)
                state[repeat2] = 1
                Z_2_folded   = ising.ising_1D_state(kT,state,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)
                Z_2_unfolded = Z_total - Z_2_folded
                
                #Calculate pairwise correlation
                state[repeat1] = 1
                Z_12_ff = ising.ising_1D_state(kT,state,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)
                
                state[repeat1] = -1
                Z_12_uf = ising.ising_1D_state(kT,state,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)  
                
                state[repeat2] = -1
                Z_12_uu = ising.ising_1D_state(kT,state,self.denaturant,G_intrin,G_inter,m_intrin,m_inter)
                
                Z_12_fu = Z_total - (Z_12_ff+Z_12_uf+Z_12_uu)
                
                self.corr_matrix[repeat1,repeat2] = (Z_12_ff-Z_12_uf-Z_12_fu+Z_12_uu)/Z_total - (Z_1_folded-Z_1_unfolded)/Z_total*(Z_2_folded-Z_2_unfolded)/Z_total
                self.corr_matrix[repeat2,repeat1] = self.corr_matrix[repeat1,repeat2]
        #Plot the correlation matrix
        py.imshow(self.corr_matrix,cmap=cm.jet,interpolation='nearest')
        py.ylim([-0.5,16.5])
        py.xlim([-0.5,16.5])
        py.xticks(nm.arange(17),nm.arange(1,18))
        py.yticks(nm.arange(17),nm.arange(1,18))
        py.colorbar()
        py.show()
    def plot_data(self,color_type='blue'):
        '''
        Plot the data
        '''
        py.plot(self.denaturant,self.exp_signal,marker='o',color=color_type,linestyle='None')
        py.title(str.format("%s at %.1f C"%(self.protein,self.temp-273.15)))
        #Put the labels
        py.xlabel('[denaturant] (M)')
        py.ylabel('Signal')
        py.show()
    
    def return_thermo(self):
        '''
        Returns the thermodynamic parameters for the protein
        '''
        G_intrin = nm.array([param.contents.value for param in self.G_intrin])
        G_inter  = nm.array([param.contents.value for param in self.G_inter])
        m_intrin = nm.array([param.contents.value for param in self.m_intrin])
        m_inter  = nm.array([param.contents.value for param in self.m_inter])
        
        return {"G_intrin":G_intrin,"G_inter":G_inter,"m_intrin":m_intrin,"m_inter":m_inter}
    
    def write_frac_folded(self,fname):
        '''
        Writes the fraction of folded as a function of denaturant concentration to a file
        '''
        f = open(fname,'w')
        for i in range(len(self.denaturant)):
            f.write("%.3f\t%.3f\n"%(self.denaturant[i],self.ave_folded[i]))
        f.close()
        
class IsingDen:
    '''
    Linear ising model with denaturant
    '''
    def __init__(self):
        self.temp           = c_double2pointer(298.15)          #Pointer to c_double
        self.kT             = c_double2pointer(_K_*298.15)      #Pointer to c_double
        self.params         = []                                #Parameter list of pointer to c_double - the values are passed to error function
        self.unique_G       = []                                #Unique parameter list for  G - only keeps the names (e.g. AB, C, etc)
        self.constr_G       = []                                #Constrained parameter list m - only keeps the names (e.g. AB, C, etc)
        self.unique_m       = []                                #Unique parameter list      G - only keeps the names (e.g. AB, C, etc)
        self.constr_m       = []                                #Constrained parameter list m - only keeps the names (e.g. AB, C, etc)
        self.data           = []                                #Array of IsingDenData
        self.G              = {}                                #Free energy dictionary - the values are pointer to c_double
        self.m              = {}                                #m-value dictionary     - the values are pointer to c_double
        self.residuals      = nm.array([])                      #Residuals
        self.topology       = nm.array([],ndmin=2)              #2D numpy array, keeps the topology of the system
        self.rank           = 0
        self.fit_counter    = 0
        
    def set_temp(self,temperature):
        '''
        Set temperature
        '''
        self.temp = c_double2pointer(temperature)
        self.kT   = c_double2pointer(_K_*temperature)
    
    def copy(self,other):
        '''
        Copy the ther isingDen object to self
        '''
        self.temp        = other.temp
        self.kT          = other.kT
        self.params      = other.params
        self.unique_G    = other.unique_G
        self.constr_G    = other.constr_G
        self.unique_m    = other.unique_m
        self.constr_m    = other.constr_m
        self.G           = other.G
        self.m           = other.m
        self.topology    = other.topology
        self.rank        = other.rank
        self.fit_counter = other.fit_counter
        self.residuals   = other.residuals + nm.zeros(len(other.residuals))
        self.data        = []
        for data in other.data:
            new_isingDenData = IsingDenData()
            new_isingDenData.copy(data)
            self.data.append(new_isingDenData)
    
    def read_data(self,ising_file,signal_type='CD'):
        '''
        Read data for ising model analysis
        '''
        f = open(ising_file,'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            data_info     = line.strip().split('\t')
            weight        = 1
            data_file     = ""
            protein_name  = ""
            if len(data_info[0]) == 0:
                continue
            if len(data_info) > 0:
                new_ising_data             = IsingDenData()
                new_ising_data.denaturant = nm.linspace(0.0,10.0,1000)
                new_ising_data.exp_signal = nm.zeros(1000)
                protein_name  = data_info[0]
            if len(data_info) > 1:
                data_file     = data_info[1]
                new_ising_data.read_data(data_file,signal_type)
            if len(data_info) > 2:
                weight        = float(data_info[2])
            
            new_ising_data.protein     = protein_name
            new_ising_data.num_repeats = len(protein_name)
            new_ising_data.weight      = weight
            new_ising_data.temp        = self.temp
            new_ising_data.kT          = self.kT
            new_ising_data.signal_type = signal_type
            
            #Decompose the protein into its intrinsic and interfacial terms
            new_ising_data.decompose_components()
            
            #Add this data to the list
            self.data.append(new_ising_data)
    
    def save_params(self,file):
        '''
        Save parameters to a file in binary format
        '''
        params = [param.contents.value for param in self.params]
        nm.save(file,params)
    
    def bootstrap(self, param_file, out_file, num_steps = 1000):
        '''
        Bootstrap the model to obtain the confidence intervals for parameters
        '''
        #First set the seed
        nm.random.seed()
        #Create a copy of the isingDen object from the current one
        bootIsingDen = IsingDen()
        bootIsingDen.copy(self)
        #First simulate data using the parameters in the param file
        self.load_params(param_file)
        self.simulate()
        #Open the file for output
        f = open(out_file,'w')
        #Write out the header
        f.write(str.format("%8s\t"%'<Chi2>'))
        for repeat_inter in self.unique_G:
            f.write(str.format("%8s\t"%('dG('+repeat_inter+')')))
        for repeat_inter in self.unique_m:
            f.write(str.format("%8s\t"%('m('+repeat_inter+')')))
        f.write('\n')
        #Now start the bootstrapping
        for step in range(num_steps):
            print str.format('Bootstrapping step: %8d'%step)
            #Go over every data we have - add noise to each data set
            for i in range(len(self.data)): 
                rand_index = nm.random.randint(0,len(self.residuals),len(self.data[i].denaturant))
                bootIsingDen.data[i].exp_signal = self.data[i].est_signal + self.residuals[rand_index]
            #Load the parameters again
            bootIsingDen.load_params(param_file)
            #Set the fit_counter to 0
            bootIsingDen.fit_counter = 0
            bootIsingDen.fit()
            #Write the parameters
            f.write(str.format("%8.3e\t"%nm.mean(bootIsingDen.residuals**2)))
            for repeat_inter in self.unique_G:
                f.write(str.format("%8.3f\t"%bootIsingDen.G[repeat_inter].contents.value))
            for repeat_inter in self.unique_m:
                f.write(str.format("%8.3f\t"%bootIsingDen.m[repeat_inter].contents.value))
            f.write('\n')
        #Close the file
        f.close()
    def load_params(self,file):
        params = nm.load(file)
        for i in range(len(self.params)):
            self.params[i].contents.value = params[i]
      
    def check_solvability(self):
        '''
        Check if we can solve the system for the worst case scenario
        '''
        self.rank  = matrix_rank(self.topology)
        num_params = len(self.unique_G)
        if self.rank < num_params:
            print 'Warning : Data set won''t probably be enough to solve the parameters'
            print str.format('Number of parameters to solve     : %d'%num_params)
            print str.format('Number of independent equations   : %d'%self.rank)
            print str.format('Number of extra constraints needed: %d'%(num_params - self.rank))
        elif self.rank > num_params:
            print 'Warning : Ising system is overdetermined'
            print str.format('Number of parameters to solve     : %d'%num_params)
            print str.format('Number of independent equations   : %d'%self.rank)
        else:
            print 'Success : Number of independent equations matches the number of parameters'
            print str.format('Number of parameters to solve     : %d'%num_params)
            print str.format('Number of independent equations   : %d'%self.rank)
    
    def normalize_data(self):
        '''
        Normalizes the data using min/max
        '''
        for ising_data in self.data:
            ising_data.exp_signal = normalize_min_max(ising_data.exp_signal)
    
    def system2params(self):
        '''
        Return the values of the parameters of the system
        '''
        params = []
        for param in self.params:
            params.append(param.contents.value)
        return nm.array(params)
    
    def params2system(self,params):
        '''
        Read the parameters to the system
        '''
        if not len(params) == len(self.params):
            print 'The number of parameters do not match the parameters of the system'
            return
        for i in range(len(params)):
            self.params[i].contents.value = params[i]
    
    def prepare_data(self):
        '''
        Prepare the dG/m-values, baseline and temperature parameters for the data
        '''
        for ising_data in self.data:
            ising_data.m_inter  = []
            ising_data.G_inter  = []
            ising_data.m_intrin = []
            ising_data.G_intrin = []
            
            #Append dummy 0 values for m_inter and g_inter
            ising_data.m_inter.append(c_double2pointer(0.0))
            ising_data.G_inter.append(c_double2pointer(0.0))
            
            #Prepare intrinsic parameters
            for repeat in ising_data.repeats:
                ising_data.G_intrin.append(self.G[repeat])
                ising_data.m_intrin.append(self.m[repeat])
            
            #Prepare interfacial parameters
            for inter in ising_data.inters:
                ising_data.G_inter.append(self.G[inter])
                ising_data.m_inter.append(self.m[inter])
            
            #Prepare the baseline parameters
            ising_data.Mf = c_double2pointer(0.0)
            ising_data.Mu = c_double2pointer(0.0)
            ising_data.Af = c_double2pointer(0.0)
            ising_data.Au = c_double2pointer(1.0)
            
            if ising_data.signal_type == 'FL':
                ising_data.Af = c_double2pointer(1.0)
                ising_data.Au = c_double2pointer(0.0)
            
            #The temperature parameters
            ising_data.temp = self.temp
            ising_data.kT   = self.kT
                
    def simulate(self,params = 'None'):
        '''
        simulate data
        '''
        #Transfer parameters to the system
        if not params == 'None':
            self.params2system(params)
        #Simulate each protein's curve
        self.residuals = []
        for ising_data in self.data:
            ising_data.simulate()
            ising_data.residuals = ising_data.est_signal - ising_data.exp_signal
            self.residuals = nm.append(self.residuals,ising_data.residuals*ising_data.weight)
        self.residuals = nm.array(self.residuals)
    
    def correlation_matrix(self,denaturant,params = 'None'):
        '''
        Build correlation matrix
        '''
        #Transfer parameters to the system
        if not params == 'None':
            self.params2system(params)
        #Simulate each protein's curve
        for ising_data in self.data:
            ising_data.correlation_matrix(nm.array([denaturant]))
            
    def errfunc(self,params):
        '''
        Error function for our ising system
        Returns the residuals
        '''
        #Iteration counter
        self.fit_counter+=1
        #Transfer parameters to the system
        self.params2system(params)
        #Simulate each protein's curve
        self.residuals = nm.array([])
        for ising_data in self.data:
            ising_data.simulate()
            ising_data.residuals = ising_data.est_signal - ising_data.exp_signal
            self.residuals       = nm.append(self.residuals, ising_data.residuals*ising_data.weight)
        if self.fit_counter % 100 == 0:
            print str.format("%s:%d\t%s:%.3e"%('Steps',self.fit_counter,'<Chi2>',nm.mean(self.residuals**2)))
        
        return self.residuals
    
    def prepare_params(self):
        '''
        Prepares the parameters(params) array for fitting
        '''
        #Parameter list
        self.params = []
        #Add free energy parameters
        for repeat_inter in self.unique_G:
            self.params.append(self.G[repeat_inter])
        
        #Add m-value parameters
        for repeat_inter in self.unique_m:
            self.params.append(self.m[repeat_inter])
    
        #Add baselines
        for ising_data in self.data:
            self.params.append(ising_data.Mf)
            self.params.append(ising_data.Af)
            self.params.append(ising_data.Mu)
            self.params.append(ising_data.Au)
    
    def fit(self,max_steps=10000,step_size=nm.finfo(float).eps):
        '''
        Fits ising model to the data provided
        '''
        init_params = nm.array([param.contents.value for param in self.params])
        best_params = nm.array([])
        best_params,success  = optimize.leastsq(self.errfunc,init_params,maxfev=max_steps,epsfcn=step_size)
        #Copy best parameters to the system
        self.params2system(best_params)
        
        return best_params
    
    def prepare_all(self, ising_file, constr_file):
        '''
        Prepare the object for ising fitting
        '''
        self.read_data(ising_file)
        self.normalize_data()
        self.prepare_system()
        self.prepare_constraints(constr_file)
        self.prepare_data()
        self.prepare_params()
        
    def plot_fit(self):
        '''
        Plot estimated values
        '''
        for ising_data in self.data:
            data_line = py.plot(ising_data.denaturant,ising_data.exp_signal,linestyle='None',marker='o')
            py.plot(ising_data.denaturant,ising_data.est_signal,color=data_line[0].get_color())
        py.show()
        
    def print_params(self):
        for repeat_inter in self.G:
            print str.format("%20s dG: %.3f\tm: %.3f"%(repeat_inter,self.G[repeat_inter].contents.value,self.m[repeat_inter].contents.value))
        
    def prepare_constraints(self,file):
        '''
        Read the constraints file
        '''
        self.constr_G = []
        self.constr_m = []
        try:
            f = open(file,'r')
            lines = f.readlines()
            f.close()
        except IOError:
            print "Sorry could not open the file"
            lines = []
            
        #Regular expressions for constraints
        G_re = re.compile('dG\((\w+)\)')
        m_re = re.compile('m\((\w+)\)')
        
        for line in lines:
            #The constraints are seperated by ;
            line_info = line.strip().split(';')
            for constraint in line_info:
                if len(constraint.strip()) > 0:
                    left,right  = constraint.strip().split('=')
                    if re.match(G_re,left.strip()):
                        left_match        = re.match(G_re,left.strip())
                        left_repeat_inter = left_match.group(1)
                        if re.match(G_re,right.strip()):
                            right_match        = re.match(G_re,right.strip())
                            right_repeat_inter = right_match.group(1)
                            #Apply the constraint
                            self.G[left_repeat_inter] = self.G[right_repeat_inter]
                            #Add the param to constrained parameters list
                            self.constr_G.append(left_repeat_inter)
                        elif is_number(right.strip()):
                            right_match = float(right.strip())
                            #Apply the constraint
                            self.G[left_repeat_inter].contents.value = right_match
                            #Add the param to constrained parameters list
                            self.constr_G.append(left_repeat_inter)
                    elif re.match(m_re,left.strip()):
                        left_match        = re.match(m_re,left.strip())
                        left_repeat_inter = left_match.group(1)
                        if re.match(m_re,right.strip()):
                            right_match        = re.match(m_re,right.strip())
                            right_repeat_inter = right_match.group(1)
                            #Apply the constraint
                            self.m[left_repeat_inter] = self.m[right_repeat_inter]
                            #Add the param to constrained parameters list
                            self.constr_m.append(left_repeat_inter)
                        elif is_number(right.strip()):
                            right_match = float(right.strip())
                            #Apply the constraint
                            self.m[left_repeat_inter].contents.value = right_match
                            #Add the param to constrained parameters list
                            self.constr_m.append(left_repeat_inter)
        
        #Prepare unique parameters list
        #Free energies
        for repeat_inter in self.G:
            if not repeat_inter in self.constr_G:
                self.unique_G.append(repeat_inter)
        #m-values
        for repeat_inter in self.m:
            if not repeat_inter in self.constr_m:
                self.unique_m.append(repeat_inter)
                
    
    def initialize_params(self,file='None'):
        '''
        Initialize the parameters
        '''
        #Average values for initial guesses
        ave_g_inter  =  10.0
        ave_g_intrin = -10.0
        ave_m_inter  =  0.5
        ave_m_intrin =  0.5
        
        #Initialize the parameters
        #Free energies
        for repeat_inter in self.unique_G:
            if len(repeat_inter) == 2:
                self.G[repeat_inter].contents.value = ave_g_inter  + nm.fabs(ave_g_inter)*(nm.random.random() - 0.5)
            else:
                self.G[repeat_inter].contents.value = ave_g_intrin + nm.fabs(ave_g_intrin)*(nm.random.random() - 0.5)
                
        #m-values
        for repeat_inter in self.unique_m:
            if len(repeat_inter) == 2:
                self.m[repeat_inter].contents.value = ave_m_inter  + nm.fabs(ave_m_inter)*(nm.random.random() - 0.5)
            else:
                self.m[repeat_inter].contents.value = ave_m_intrin + nm.fabs(ave_m_intrin)*(nm.random.random() - 0.5)
        
        #Now read from the file
        if not file == 'None':
            try:
                f = open(file,'r')
                lines = f.readlines()
                f.close()
            except IOError:
                print "Sorry could not open the file: "+file
                lines = []
        else:
            lines = []
            
        #Regular expressions for constraints
        G_re = re.compile('dG\((\w+)\)')
        m_re = re.compile('m\((\w+)\)')
        
        for line in lines:
            #The constraints are seperated by ;
            line_info = line.strip().split(';')
            for constraint in line_info:
                if len(constraint.strip()) > 0:
                    left,right  = constraint.strip().split('=')
                    if re.match(G_re,left.strip()):
                        left_match        = re.match(G_re,left.strip())
                        left_repeat_inter = left_match.group(1)
                        if is_number(right.strip()) and left_repeat_inter in self.unique_G:
                            right_match = float(right.strip())
                            #Initialize the parameter value
                            self.G[left_repeat_inter].contents.value = right_match
                    elif re.match(m_re,left.strip()):
                        left_match        = re.match(m_re,left.strip())
                        left_repeat_inter = left_match.group(1)
                        if is_number(right.strip()) and left_repeat_inter in self.unique_G:
                            right_match = float(right.strip())
                            #Apply the constraint
                            self.m[left_repeat_inter].contents.value = right_match

    
    def prepare_system(self):
        '''
        After reading the data prepare the parameters/topology matrix of the model
        '''
        #Read the repeats/interfaces - and prepare baselines
        for ising_data in self.data:
            for repeat in ising_data.repeats:
                if not repeat in self.G:
                    self.G[repeat] = c_double2pointer(0.0)
                    self.m[repeat] = c_double2pointer(0.0)
            for inter in ising_data.inters:
                if not inter in self.G:
                    self.G[inter] = c_double2pointer(0.0)
                    self.m[inter] = c_double2pointer(0.0)
        
        #Build topology matrix
        self.topology = nm.array([],ndmin=2)
        for ising_data in self.data:
            row_protein = []
            for param in self.G:
                if len(param) == 2:
                    row_protein.append(ising_data.inters.count(param))
                else:
                    row_protein.append(ising_data.repeats.count(param))
            #If the second dimension size is 0 : empty 2D array
            if self.topology.shape[1] == 0:
                self.topology = nm.array(row_protein,ndmin=2)
            else:
                self.topology = nm.append(self.topology,nm.array(row_protein,ndmin=2),axis=0)
    
    

class TwoStateDen:
    '''
    Two state folding model
    '''
    def __init__(self):
        self.temp       = 298.15
        self.denaturant = []
        self.signal     = []
        self.dG         = 0
        self.m_value    = 0
        self.Mf         = 0
        self.Mu         = 0
        self.Af         = 0
        self.Au         = 0
        self.chi2       = 0
        self.residuals  = []
        
    def simulate(self):
        '''
        Simulates a 2-state titration
        '''
        kT = self.temp*k
        
        frac_folded   = 1/(1+nm.exp(-(self.dG - self.m_value*self.denaturant)/kT))
        frac_unfolded = 1 - frac_folded
        
        self.signal = (self.Mf*self.denaturant+self.Af)*frac_folded+(self.Mu*self.denaturant+self.Au)*frac_unfolded
        
    def set_temp(self,temperature):
        '''
        Set the temperature for two state model
        Temperature has to be in Kelvin units
        '''
        self.temp = temperature
    
    def errfunc(self,params):
        '''
        Error function for fitting 2-state model to titration data
        '''
        test_model = TwoStateDen()
        test_model.denaturant = self.denaturant
        test_model.temp    = self.temp
        test_model.dG      = params[0]
        test_model.m_value = params[1]
        test_model.Mf      = params[2]
        test_model.Af      = params[3]
        test_model.Mu      = params[4]
        test_model.Au      = params[5]
        
        #Simulate data
        test_model.simulate()
        #Return the residuals
        return self.signal - test_model.signal
    
    def copy_model(self,other):
        self.temp       = other.temp
        self.denaturant = other.denaturant
        self.signal     = other.signal
        self.dG         = other.dG
        self.m_value    = other.m_value
        self.Mf         = other.Mf
        self.Mu         = other.Mu
        self.Af         = other.Af
        self.Au         = other.Au
        self.chi2       = other.chi2
        self.residuals  = other.residuals
        
    def correct_with_baselines(self):
        '''
        Corrects titration data using baseline values
        '''
        return (self.signal - (self.Mf*self.denaturant+self.Af))/((self.Mu-self.Mf)*self.denaturant+(self.Au-self.Af))
    
    def plot_fit(self,color_type='blue',title=''):
        '''
        Plot the fitted curve with the raw data
        '''
        fit_model = TwoStateDen()
        fit_model.copy_model(self)
        fit_model.denaturant = nm.linspace(nm.min(fit_model.denaturant),nm.max(fit_model.denaturant),100)
        fit_model.simulate()
        
        #Correct the data and fitted curve with baselines
        self.signal      = self.correct_with_baselines()
        fit_model.signal = fit_model.correct_with_baselines() 
        #Plot data
        py.plot(self.denaturant,self.signal,marker='o',color=color_type,linestyle='None')
        #Plot fitted curve
        py.plot(fit_model.denaturant,fit_model.signal,color=color_type,linestyle='-',linewidth=3)
        #Put the title
        py.title(title)
        #Put the labels
        py.xlabel('[denaturant] (M)')
        py.ylabel('Fraction of unfolded')
        #Print the parameters found
        py.text(nm.min(self.denaturant),0.8,r'$\Delta G^{0}_{u}$  =%5.2f kcal/mol'%self.dG,fontsize=18)
        py.text(nm.min(self.denaturant),0.7,r'$m-val$=%5.2f kcal/mol.M'%self.m_value,fontsize=18)
        py.show()
    
    def fit(self,init_params = [1.0,0.5,0.1,0.0,0.1,1.0]):
        '''
        Fits 2-state model to a chemical denaturation data
        returns free energy of unfolding(dG) and m-value
        '''
        #Normalize the data by min/max
        self.signal = normalize_min_max(self.signal)
        
        #The return from leastsq fitting routine
        #If 1 -> successful fitting
        success = 0
        
        while success != 1:
            params  = init_params
            params[0:2] += nm.random.rand(2)*nm.array([10,10])
            best_params,success  = optimize.leastsq(self.errfunc,params)
        
        #Get the thermodynamic parameters of interest
        self.dG      = best_params[0]
        self.m_value = best_params[1]
        self.Mf      = best_params[2]
        self.Af      = best_params[3]
        self.Mu      = best_params[4]
        self.Au      = best_params[5]
        
        #Plot the fitted curve with the data
        self.plot_fit()

class Titration:
    '''
    Titration Data class
    '''
    def __init__(self):
       self.temp        = 298.15
       self.titrant     = nm.array([])
       self.cd_signal   = nm.array([])
       self.cd_error    = nm.array([])
       self.cd_dynode   = nm.array([])
       self.fl_signal   = nm.array([])
       self.fl_dynode   = nm.array([])
       self.inj_vol     = nm.array([])
       self.jacket_temp = nm.array([])
    
    def fit_2state(self,signal_type='CD',init_params = [1.0,0.5,0.1,0.0,0.1,1.0]):
        model = TwoStateDen()
        model.set_temp(self.temp)
        model.denaturant = self.titrant
        model.signal = self.cd_signal
        if signal_type == 'FL':
            model.signal = self.fl_signal
        model.fit(init_params = init_params)
        return [model.dG,model.m_value]
        
    def plot_data(self,signal_type='CD',color_type='blue'):
        '''
        Plot CD/FL data
        '''
        den    = self.titrant
        signal = self.cd_signal
        if signal_type == 'FL':
            signal = self.fl_signal
        py.plot(den,signal,marker='o',color=color_type,linestyle='None')
        py.title(signal_type+' signal')
        #Put the labels
        py.xlabel('[denaturant] (M)')
        py.ylabel('Fraction of unfolded')
        py.show()
        
    def read_data(self,file):
        '''
        Function that reads AVIV titration data:
        col_names keeps the column names
        titration_data keeps the data
        '''
        titration_data = []
        col_names = []
        try:
            f = open(file,'r')
        except IOError:
            print 'Titration data not found'
            return
        lines = f.readlines()
        f.close()
        title_re = re.compile('\sX')
        data_re  = re.compile('\d')
        for line in lines:
            if re.match(title_re,line):
                col_names = line.strip().split()
            elif re.match(data_re,line):
                data_row = [float(data) for data in line.strip().split()]
                titration_data.append(data_row)
            elif line.strip() == '$ENDDATA':
                break
        titration_data = nm.array(titration_data)
        
        #Print the file name to be processed
        print 'Reading '+ file 
        
        #Assign the titration variables
        
        #Titrant concentrations
        try:
            index = col_names.index('X')
            self.titrant = nm.array(titration_data[:,index])
        except ValueError:
            print 'Titrant concentration is not available'
        
        #CD signal
        try:
            index = col_names.index('CD_Signal')
            self.cd_signal = nm.array(titration_data[:,index])
        except ValueError:
            print 'CD signal is not available'
        
        #CD signal error
        try:
            index = col_names.index('Error')
            self.cd_error = nm.array(titration_data[:,index])
        except ValueError:
            print 'CD signal error is not available'
        
        #CD dynode
        try:
            index = col_names.index('Dynode')
            self.cd_dynode = nm.array(titration_data[:,index])
        except ValueError:
            print 'CD dynode is not available'
            
        #FL signal
        try:
            index = col_names.index('FL_Signal')
            self.fl_signal = nm.array(titration_data[:,index])
        except ValueError:
            print 'FL signal is not available'
        
        #FL dynode
        try:
            index = col_names.index('FL_Dynode')
            self.fl_dynode = nm.array(titration_data[:,index])
        except ValueError:
            print 'FL dynode is not available'
        
        #Injection volumes
        try:
            index = col_names.index('Inj._Vol._ul.')
            self.inj_vol = nm.array(titration_data[:,index])
        except ValueError:
            print 'Injection volumes is not available'
        
        #Jacket temperature
        try:
            index = col_names.index('Jacket_Temp.')
            self.jacket_temp = nm.array(titration_data[:,index])
        except ValueError:
            print 'Jacket temperature is not available'

    
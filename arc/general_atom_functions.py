# -*- coding: utf-8 -*-
"""
Implements general single-atom calculations

This module calculates single (isolated) atom properties of all alkali metals in
general. For example, it calculates dipole matrix elements, quandrupole matrix
elements, etc.  Also, some helpful general functions are here, e.g. for saving
and loading calculations (single-atom and pair-state based), printing state
labels etc.


"""

from __future__ import division, print_function, absolute_import

from math import exp,log,sqrt
# for web-server execution, uncomment the following two lines
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import shutil

from .wigner import Wigner6j,Wigner3j,wignerD,CG,wignerDmatrix
from scipy.constants import physical_constants, pi , epsilon_0, hbar
from scipy.constants import k as C_k
from scipy.constants import c as C_c
from scipy.constants import h as C_h
from scipy.constants import e as C_e
from scipy.constants import m_e as C_m_e
from scipy.optimize import curve_fit
# for matrices
from numpy import zeros,savetxt, complex64,complex128
from numpy.linalg import eigvalsh,eig,eigh
from numpy.ma import conjugate
from numpy.lib.polynomial import real

from scipy.sparse import csr_matrix
from scipy.sparse import kron as kroneckerp
from scipy.sparse.linalg import eigsh
from scipy.special.specfun import fcoef
from scipy import floor

import sys, os
if sys.version_info > (2,):
    xrange = range

try:
    import cPickle as pickle   # fast, C implementation of the pickle
except:
    import pickle   # Python 3 already has efficient pickle (instead of cPickle)
import gzip
import csv
import sqlite3
import mpmath
sqlite3.register_adapter(np.float64, float)
sqlite3.register_adapter(np.float32, float)
sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)

DPATH = os.path.join(os.path.expanduser('~'), '.arc-data')

def setup_data_folder():
    """ Setup the data folder in the users home directory.

    """
    if not os.path.exists(DPATH):
        os.makedirs(DPATH)
        dataFolder = os.path.join(os.path.dirname(os.path.realpath(__file__)),"data")
        for fn in os.listdir(dataFolder):
            if os.path.isfile(os.path.join(dataFolder, fn)):
                shutil.copy(os.path.join(dataFolder, fn), DPATH)
class Atom(object):
    """
        Implements general calculations for alkali atoms.

        This abstract class implements general calculations methods.

        Args:
            preferQuantumDefects (bool):
                Use quantum defects for energy level calculations. If False,
                uses NIST ASD values
                where available. If True, uses quantum defects for energy calculations
                for principal quantum numbers equal or above
                :obj:`minQuantumDefectN` which is specified for each element
                separately. For principal quantum numbers below this value,
                NIST ASD values are used, since quantum defects don't reproduce
                well low-lying states. Default is True.
            cpp_numerov (bool):
                should the wavefunction be calculated with Numerov algorithm
                implemented in C++; if False, it uses pure Python implementation
                that is much slower. Default is True.

    """
    # ALL PARAMETERS ARE IN ATOMIC UNITS (Hatree)
    alpha = physical_constants["fine-structure constant"][0]

    alphaC = 0.0    #: Core polarizability
    Z = 0.0       #: Atomic number

    # state energies from NIST values
    # sEnergy [n,l] = state energy for n, l, j = l-1/2
    # sEnergy [l,n] = state energy for j = l+1/2
    sEnergy = 0
    NISTdataLevels = 0
    scaledRydbergConstant = 0 #: in eV

    
    """ Contains list of modified Rydberg-Ritz coefficients for calculating
        quantum defects for [[ :math:`S_{1/2},P_{1/2},D_{3/2},F_{5/2}`],
        [ :math:`S_{1/2},P_{3/2},D_{5/2},F_{7/2}`]]."""

    levelDataFromNIST = ""                  #: location of stored NIST values of measured energy levels in eV
    dipoleMatrixElementFile = ""            #: location of hard-disk stored dipole matrix elements
    quadrupoleMatrixElementFile = ""        #: location of hard-disk stored dipole matrix elements

    dataFolder = DPATH

    # now additional literature sources of dipole matrix elements

    literatureDMEfilename = ""
    """
        Filename of the additional literature source values of dipole matrix
        elements.

        These additional values should be saved as reduced dipole matrix elements
        in J basis.

    """


    #: levels that are for smaller principal quantum number (n) than ground level, but are above in energy due to angular part
    extraLevels = []

    #: principal quantum number for the ground state
    groundStateN = 0

    #: swich - should the wavefunction be calculated with Numerov algorithm implemented in C++
    cpp_numerov = True

    mass = 0.  #: atomic mass in kg
    abundance = 1.0  #: relative isotope abundance

    elementName = "elementName"  #: Human-readable element name

    preferQuantumDefects = False
    minQuantumDefectN = 0  #: minimal quantum number for which quantum defects can be used; uses measured energy levels otherwise

    semi= True
    # SQLite connection and cursor
    conn = False
    c = False

    def __init__(self, preferQuantumDefects=True,cpp_numerov=True):

        self.cpp_numerov = cpp_numerov
        self.preferQuantumDefects = preferQuantumDefects

        self._databaseInit()

        if self.cpp_numerov:
            from .arc_c_extensions import NumerovWavefunction
            self.NumerovWavefunction = NumerovWavefunction


        # load dipole matrix elements previously calculated
        data=[]
        if (self.dipoleMatrixElementFile != ""):
            if (preferQuantumDefects == False):
                self.dipoleMatrixElementFile  = "NIST_"+self.dipoleMatrixElementFile

            try:
                data = np.load(os.path.join(self.dataFolder,\
                                            self.dipoleMatrixElementFile),\
                               encoding = 'latin1')
                #print(data)
            except IOError as e:
                print("Error reading dipoleMatrixElement File "+\
                    os.path.join(self.dataFolder,self.dipoleMatrixElementFile))
                print(e)
        # save to SQLite database
        #print(data[0])
        try:
            self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='dipoleME';''')
            if (self.c.fetchone()[0] == 0):
                # create table
                self.c.execute('''CREATE TABLE IF NOT EXISTS dipoleME
                 (n1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
                 n2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
                 dme DOUBLE, s TINYINT UNSIGNED, semi TINYINT UNSIGNED,
                 PRIMARY KEY (n1,l1,j1_x2,n2,l2,j2_x2,s,semi)
                ) ''')
                if (len(data)>0):
                    self.c.executemany('INSERT INTO dipoleME VALUES (?,?,?,?,?,?,?,?,?)', data)
                self.conn.commit()
        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database")
            print(e)
            exit()

        # load quadrupole matrix elements previously calculated
        data=[]
        if (self.quadrupoleMatrixElementFile != ""):
            if (preferQuantumDefects == False):
                self.quadrupoleMatrixElementFile  = "NIST_"+self.quadrupoleMatrixElementFile
            try:
                data = np.load(os.path.join(self.dataFolder,\
                                            self.quadrupoleMatrixElementFile),\
                               encoding = 'latin1')

            except IOError as e:
                print("Error reading quadrupoleMatrixElementFile File "+\
                    os.path.join(self.dataFolder,self.quadrupoleMatrixElementFile))
                print(e)
        # save to SQLite database
        try:
            self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='quadrupoleME';''')
            if (self.c.fetchone()[0] == 0):
                # create table
                self.c.execute('''CREATE TABLE IF NOT EXISTS quadrupoleME
                 (n1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
                 n2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
                 qme DOUBLE, s TINYINT UNSIGNED, semi TINYINT UNSIGNED,
                 PRIMARY KEY (n1,l1,j1_x2,n2,l2,j2_x2,s,semi)
                ) ''')
                if (len(data)>0):
                    self.c.executemany('INSERT INTO quadrupoleME VALUES (?,?,?,?,?,?,?,?,?)', data)
                self.conn.commit()
        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database")
            print(e)
            exit()


        return
    def _databaseInit(self):
   
        self.conn = sqlite3.connect(os.path.join(self.dataFolder,\
                                                 self.precalculatedDB))
        self.c = self.conn.cursor()
        return

    def getPressure(self,temperature):
        """ Vapour pressure (in Pa) at given temperature

            Args:
                temperature (float): temperature in K
            Returns:
                float: vapour pressure in Pa
        """
        print("Error: getPressure to-be implement in child class (otherwise this\
                call is invalid for the specified atom")
        exit()

    def getNumberDensity(self,temperature):
        """ Atom number density at given temperature

            See `calculation of basic properties example snippet`_.

            .. _`calculation of basic properties example snippet`:
                ./Rydberg_atoms_a_primer.html#General-atomic-properties

            Args:
                temperature (float): temperature in K
            Returns:
                float: atom concentration in :math:`1/m^3`
        """
        return self.getPressure(temperature)/(C_k*temperature)

    def getAverageInteratomicSpacing(self,temperature):
        """
            Returns average interatomic spacing in atomic vapour

            See `calculation of basic properties example snippet`_.

            .. _`calculation of basic properties example snippet`:
                ./Rydberg_atoms_a_primer.html#General-atomic-properties

            Args:
                temperature (float): temperature of the atomic vapour

            Returns:
                float: average interatomic spacing in m
        """
        return  (5./9.)*self.getNumberDensity(temperature)**(-1./3.)

    def radialWavefunction(self,l,s,j,stateEnergy,innerLimit,outerLimit,step):
        """
        Radial part of electron wavefunction

        Calculates radial function with Numerov (from outside towards the core).
        Note that wavefunction might not be calculated all the way to the requested
        `innerLimit` if the divergence occurs before. In that case third returned
        argument gives nonzero value, corresponding to the first index in the array
        for which wavefunction was calculated. For quick example see
        `Rydberg wavefunction calculation snippet`_.

        .. _`Rydberg wavefunction calculation snippet`:
            ./Rydberg_atoms_a_primer.html#Rydberg-atom-wavefunctions



        Args:
            l (int): orbital angular momentum
            s (float): spin angular momentum
            j (float): total angular momentum
            stateEnergy (float): state energy, relative to ionization threshold,
                should be given in atomic units (Hatree)
            innerLimit (float): inner limit at which wavefunction is requested
            outerLimit (float): outer limit at which wavefunction is requested
            step (flaot): radial step for integration mesh (a.u.)
        Returns:
            List[float], List[flaot], int:
                :math:`r`

                :math:`R(r)\cdot r`

        .. note::
            Radial wavefunction is not scaled to unity! This normalization
            condition means that we are using spherical harmonics which are
            normalized such that
            :math:`\\int \\mathrm{d}\\theta~\\mathrm{d}\\psi~Y(l,m_l)^* \\times \
            Y(l',m_{l'})  =  \\delta (l,l') ~\\delta (m_l, m_{l'})`.

        Note:
            Alternative calculation methods can be added here (potenatial
            package expansion).

        """
        innerLimit = max(4. * step, innerLimit)  # prevent divergence due to hitting 0
        if self.cpp_numerov:
            # efficiant implementation in C
            if (l<4):
                d = self.NumerovWavefunction(innerLimit,outerLimit,\
                                        step,0.01,0.01,\
                                        l,s,j,stateEnergy,self.alphaC,self.alpha,\
                                        self.Z,
                                        self.a1[l],self.a2[l],self.a3[l],self.a4[l],\
                                        self.rc[l],\
                                        (self.mass-C_m_e)/self.mass)
            else:
                d = self.NumerovWavefunction(innerLimit,outerLimit,\
                                        step,0.01,0.01,\
                                        l,s,j,stateEnergy,self.alphaC,self.alpha,\
                                        self.Z,0.,0.,0.,0.,0.,\
                                        (self.mass-C_m_e)/self.mass)

            psi_r  = d[0]
            r = d[1]
            suma = np.trapz(psi_r**2, x=r)
            psi_r = psi_r/(sqrt(suma))
        else:
            # full implementation in Python
            mu = (self.mass-C_m_e)/self.mass
            def potential(x):
                r = x*x
                #return 2.*mu*(stateEnergy-self.potential(l, s, j, r))-l*(l+1)/(r**2)
                return -3./(4.*r)+4.*r*(\
                      2.*mu*(stateEnergy-self.potential(l, s, j, r))-l*(l+1)/(r**2)\
                    )
            #print(outerLimit)
            #print(innerLimit)
            r,psi_r = NumerovBack(innerLimit,outerLimit,potential,\
                                         step,0.01,0.01)

            suma = np.trapz(psi_r**2, x=r)
            psi_r = psi_r/(sqrt(suma))

        return r,psi_r

    def getTransitionWavelength(self,n1,l1,j1,n2,l2,j2, s=0.5):
        """
            Calculated transition wavelength (in vacuum) in m.

            Returned values is given relative to the centre of gravity of the
            hyperfine-split states.

            Args:
                n1 (int): principal quantum number of the state **from** which we are going
                l1 (int): orbital angular momentum of the state **from** which we are going
                j1 (float): total angular momentum of the state **from** which we are going
                n2 (int): principal quantum number of the state **to** which we are going
                l2 (int): orbital angular momentum of the state **to** which we are going
                j2 (float): total angular momentum of the state **to** which we are going

            Returns:
                float:
                    transition wavelength (in m). If the returned value is negative,
                    level from which we are going is **above** the level to which we are
                    going.
        """

        return (C_h*C_c)/((self.getEnergy(n2, l2, j2, s)-self.getEnergy(n1, l1, j1, s))*C_e)

    def getTransitionFrequency(self,n1,l1,j1,n2,l2,j2,s=0.5):
        """
            Calculated transition frequency in Hz

            Returned values is given relative to the centre of gravity of the
            hyperfine-split states.

            Args:
                n1 (int): principal quantum number of the state **from** which we are going
                l1 (int): orbital angular momentum of the state **from** which we are going
                j1 (float): total angular momentum of the state **from** which we are going
                n2 (int): principal quantum number of the state **to** which we are going
                l2 (int): orbital angular momentum of the state **to** which we are going
                j2 (float): total angular momentum of the state **to** which we are going

            Returns:
                float:
                    transition frequency (in Hz). If the returned value is negative,
                    level from which we are going is **above** the level to which we are
                    going.
        """
        return (self.getEnergy(n2, l2, j2,s)-self.getEnergy(n1, l1, j1,s))*C_e/C_h
    def getRadialMatrixElement(self,n1,l1,j1,n2,l2,j2,s= 0.5,useLiterature=True):
        if (n1 == n2 and l1 == l2 and j1 == j2):
            #if we are dipole coupling the same states then we return 0
            return 0
        else:
            if self.semi ==False:
            
                """
                    Radial part of the dipole matrix element
        
                    Calculates :math:`\\int \\mathbf{d}r~R_{n_1,l_1,j_1}(r)\cdot \
                        R_{n_1,l_1,j_1}(r) \cdot r^3`.
        
                    Args:
                        n1 (int): principal quantum number of state 1
                        l1 (int): orbital angular momentum of state 1
                        j1 (float): total angular momentum of state 1
                        n2 (int): principal quantum number of state 2
                        l2 (int): orbital angular momentum of state 2
                        j2 (float): total angular momentum of state 2
        
                    Returns:
                        float: dipole matrix element (:math:`a_0 e`).
                """
                semi = 0
                
                dl = abs(l1-l2)
                dj = abs(j2-j2)
                if not(dl==1 and (dj<1.1)):
                    return 0
        
                if (self.getEnergy(n1, l1, j1,s)>self.getEnergy(n2, l2, j2,s)):
                    temp = n1
                    n1 = n2
                    n2 = temp
                    temp = l1
                    l1 = l2
                    l2 = temp
                    temp = j1
                    j1 = j2
                    j2 = temp
        
                n1 = int(n1)
                n2 = int(n2)
                l1 = int(l1)
                l2 = int(l2)
                j1_x2 = int(round(2*j1))
                j2_x2 = int(round(2*j2))
        
                #if useLiterature:
                    # is there literature value for this DME? If there is, use the best one (smalles error)
                #    self.c.execute('''SELECT dme FROM literatureDME WHERE
                #     n1= ? AND l1 = ? AND j1_x2 = ? AND
                #     n2 = ? AND l2 = ? AND j2_x2 = ? AND
                #     s = ?
                #     ORDER BY errorEstimate ASC''',(n1,l1,j1_x2,n2,l2,j2_x2,s))
                #    answer = self.c.fetchone()
                #    if (answer):
                        # we did found literature value
                #        return answer[0]
        
        
                # was this calculated before? If it was, retrieve from memory
                self.c.execute('''SELECT dme FROM dipoleME WHERE
                 n1= ? AND l1 = ? AND j1_x2 = ? AND
                 n2 = ? AND l2 = ? AND j2_x2 = ? AND
                 s = ? AND semi = ?''',(n1,l1,j1_x2,n2,l2,j2_x2,s,semi))
                dme = self.c.fetchone()
                if (dme):
                    return dme[0]
        
                #LIZZY TEMP
                step = 0.001
                r1,psi1_r1 = self.radialWavefunction(l1,s,j1,\
                                                       self.getEnergy(n1, l1, j1,s)/27.211,\
                                                       self.alphaC**(1/3.0),\
                                                        2.0*n1*(n1+15.0), step)
                r2,psi2_r2 = self.radialWavefunction(l2,s,j2,\
                                                       self.getEnergy(n2, l2, j2,s)/27.211,\
                                                       self.alphaC**(1/3.0),\
                                                        2.0*n2*(n2+15.0), step)
        
                upTo = min(len(r1),len(r2))
        
                # note that r1 and r2 change in same staps, starting from the same value
                dipoleElement = np.trapz(np.multiply(np.multiply(psi1_r1[0:upTo],psi2_r2[0:upTo]),\
                                                   r1[0:upTo]), x = r1[0:upTo])
        
                self.c.execute(''' INSERT INTO dipoleME VALUES (?,?,?, ?,?,?, ?,?,?)''',\
                               [n1,l1,j1_x2,n2,l2,j2_x2, dipoleElement,s,semi] )
                self.conn.commit()
        
                return dipoleElement
            else:
                semi = 1
                #Lizzy get rid of this
               #useLiterature = False
                if (self.getEnergy(n1, l1, j1,s)>self.getEnergy(n2, l2, j2,s)):
                    temp = n1
                    n1 = n2
                    n2 = temp
                    temp = l1
                    l1 = l2
                    l2 = temp
                    temp = j1
                    j1 = j2
                    j2 = temp
        
                n1 = int(n1)
                n2 = int(n2)
                l1 = int(l1)
                l2 = int(l2)
                j1_x2 = int(round(2*j1))
                j2_x2 = int(round(2*j2))
                s = int(s)



                if useLiterature:
                    #print('got in the if')
                    # is there literature value for this DME? If there is, use the best one (smalles error)
                    self.c.execute('''SELECT dme FROM literatureDME WHERE
                     n1= ? AND l1 = ? AND j1_x2 = ? AND
                     n2 = ? AND l2 = ? AND j2_x2 = ? AND
                     s = ?
                     ORDER BY errorEstimate ASC''',(n1,l1,j1_x2,n2,l2,j2_x2,s))

                    answer = self.c.fetchone()
                
                    if (answer):
                        # we did found literature value
                        return answer[0]
        
        
                # was this calculated before? If it was, retrieve from memory
                self.c.execute('''SELECT dme FROM dipoleME WHERE
                 n1= ? AND l1 = ? AND j1_x2 = ? AND
                 n2 = ? AND l2 = ? AND j2_x2 = ? AND
                 s = ? AND semi = ?''',(n1,l1,j1_x2,n2,l2,j2_x2,s,semi))
                dme = self.c.fetchone()
                if (dme):
                    #print('from dme')
                    #if (l1 ==1): print('Fetch')

                    return dme[0]
                #print('Using the semiclassical approch to calculating the Radial Matrix element!')
                dipoleElement = self.getRadialMatrixElementSemiClassical(n1,l1,j1, n2, l2, j2, s)
                
                self.c.execute(''' INSERT INTO dipoleME VALUES (?,?,?, ?,?,?, ?,?,?)''',\
                           [n1,l1,j1_x2,n2,l2,j2_x2, float(dipoleElement),s,semi] )
                self.conn.commit()
                return dipoleElement
                    

    def getRadialMatrixElementSemiClassical(self,n,l,j,n1,l1,j1, s=0.5):
        #get the effective principal number of both states
        nu = n - self.getQuantumDefect(n,l,j,s)
        nu1 = n1 - self.getQuantumDefect(n1,l1,j1,s)
        

        #get the parameters required to calculate the sum
        l_c = (l+l1+1.)/2.
        nu_c = sqrt(nu*nu1)

        delta_nu = nu- nu1
        delta_l = l1 -l
       
        #I am not sure if this correct 
        
        gamma  = (delta_l*l_c)/nu_c

        if delta_nu ==0:
            g0 = 1
            g1 = 0
            g2 = 0
            g3 = 0
        else:

            g0 = (1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) - mpmath.angerj(delta_nu+1,-delta_nu))
            g1 = -(1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) + mpmath.angerj(delta_nu+1,-delta_nu))
            g2 = g0 - mpmath.sin(pi*delta_nu)/(pi*delta_nu)
            g3 = (delta_nu/2.)*g0 + g1

        radial_ME = (3/2)*nu_c**2*(1-(l_c/nu_c)**(2))**0.5*(g0 + gamma*g1 + gamma**2*g2 + gamma**3*g3)
        return float(radial_ME)

    def getQuadrupoleMatrixElementSemiClassical(self,n,l,j,n1,l1,j1, s=0.5):#
        dl = abs(l1-l)

        
        nu = n - self.getQuantumDefect(n,l,j,s)
        nu1 = n1 - self.getQuantumDefect(n1,l1,j1,s)

        #get the parameters required to calculate the sum
        l_c = (l+l1+1.)/2.
        nu_c = sqrt(nu*nu1)

        delta_nu = nu- nu1
        delta_l = l1 -l

        gamma  = (delta_l*l_c)/nu_c

        if delta_nu ==0:
            q = np.array([1,0,0,0])
        else:
        
            g0 = (1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) - mpmath.angerj(delta_nu+1,-delta_nu))
            g1 = -(1./(3.*delta_nu))*(mpmath.angerj(delta_nu-1.,-delta_nu) + mpmath.angerj(delta_nu+1,-delta_nu))
        
            q = np.zeros((4,))
            q[0] = -(6./(5.*delta_nu))*g1
            q[1] = -(6./(5.*delta_nu))*g0 + (6./5.)*np.sin(pi*delta_nu)/(pi*delta_nu**2)
            q[2] = -(3./4.)*(6./(5.*delta_nu) *g1 +g0)
            q[3] = 0.5*(delta_nu*0.5*q[0] +q[1])
            
        sm = 0
        
        if dl ==0:
            quadrupoleElement = (5./2.)*nu_c**4*(1.-(3.*l_c**2)/(5*nu_c**2))
            for p in range(0,2,1):
                sm += gamma**(2*p)*q[2*p]
            return quadrupoleElement*sm
        
        elif dl == 2:
            quadrupoleElement = (5./2.)*nu_c**4*(1-(l_c +1)**2/(nu_c**2))**0.5* (1-(l_c +2)**2/(nu_c**2))**0.5
            for p in range(0,4):
                sm += gamma**(p)*q[p]
            return quadrupoleElement*sm
        else:
            return 0

    def getQuadrupoleMatrixElement(self,n1,l1,j1,n2,l2,j2,s = 0.5):
        """
            Radial part of the quadrupole matrix element

            Calculates :math:`\\int \\mathbf{d}r~R_{n_1,l_1,j_1}(r)\cdot \
            R_{n_1,l_1,j_1}(r) \cdot r^4`.
            See `Quadrupole calculation example snippet`_  .

            .. _`Quadrupole calculation example snippet`:
                ./Rydberg_atoms_a_primer.html#Quadrupole-matrix-elements

            Args:
                n1 (int): principal quantum number of state 1
                l1 (int): orbital angular momentum of state 1
                j1 (float): total angular momentum of state 1
                n2 (int): principal quantum number of state 2
                l2 (int): orbital angular momentum of state 2
                j2 (float): total angular momentum of state 2

            Returns:
                float: quadrupole matrix element (:math:`a_0^2 e`).
        """
        dl = abs(l1-l2)
        dj = abs(j1-j2)
        if self.semi == False:
            if not ((dl==0 or dl==2 or dl==1)and (dj<2.1)):
                return 0
    
            if (self.getEnergy(n1, l1, j1,s)>self.getEnergy(n2, l2, j2,s)):
                temp = n1
                n1 = n2
                n2 = temp
                temp = l1
                l1 = l2
                l2 = temp
                temp = j1
                j1 = j2
                j2 = temp
    
            n1 = int(n1)
            n2 = int(n2)
            l1 = int(l1)
            l2 = int(l2)
            j1_x2 = int(round(2*j1))
            j2_x2 = int(round(2*j2))
    
            # was this calculated before? If yes, retrieve from memory.
            self.c.execute('''SELECT qme FROM quadrupoleME WHERE
             n1= ? AND l1 = ? AND j1_x2 = ? AND
             n2 = ? AND l2 = ? AND j2_x2 = ? AND s = ?''',(n1,l1,j1_x2,n2,l2,j2_x2,s))
            qme = self.c.fetchone()
            if (qme):
                return qme[0]
    
            # if it wasn't, calculate now
            step = 0.001
            r1, psi1_r1 = self.radialWavefunction(l1,s,j1,\
                                                   self.getEnergy(n1, l1, j1,s)/27.211,\
                                                   self.alphaC**(1/3.0), \
                                                   2.0*n1*(n1+15.0), step)
            r2, psi2_r2 = self.radialWavefunction(l2,s,j2,\
                                                   self.getEnergy(n2, l2, j2,s)/27.211,\
                                                   self.alphaC**(1/3.0), \
                                                   2.0*n2*(n2+15.0), step)
    
            upTo = min(len(r1),len(r2))
    
            # note that r1 and r2 change in same staps, starting from the same value
            quadrupoleElement = np.trapz(np.multiply(np.multiply(psi1_r1[0:upTo],psi2_r2[0:upTo]),\
                                                   np.multiply(r1[0:upTo],r1[0:upTo])),\
                                         x = r1[0:upTo])
    
    
    
            self.c.execute(''' INSERT INTO quadrupoleME VALUES (?,?,?, ?,?,?, ?,?)''',\
                           [n1,l1,j1_x2,n2,l2,j2_x2, quadrupoleElement,s] )
            self.conn.commit()
    
            return quadrupoleElement
        else: 
            return self.getQuadrupoleMatrixElementSemiClassical(n1,l1,j1,n2,l2,j2,s)
    
    def getC6term(self,n,l,j, n1,l1,j1, n2,l2,j2, m,mm,m1, m2,q1,q2,semi):
        """
            C6 interaction term for the given two pair-states

            Calculates :math:`C_6` intaraction term for :math:`|n,l,j,n,l,j\\rangle\
            \\leftrightarrow |n_1,l_1,j_1,n_2,l_2,j_2\\rangle`. For details
            of calculation see Ref. [#c6r1]_.

            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  :math:`C_6 = \\frac{1}{4\\pi\\varepsilon_0} \
                    \\frac{|\\langle n,l,j |er|n_1,l_1,j_1\\rangle|^2|\
                    \\langle n,l,j |er|n_2,l_2,j_2\\rangle|^2}\
                    {E(n_1,l_1,j_2,n_2,j_2,j_2)-E(n,l,j,n,l,j)}`
                (:math:`h` Hz m :math:`{}^6`).

            Example:
                We can reproduce values from Ref. [#c6r1]_ for C3 coupling
                to particular channels. Taking for example channels described
                by the Eq. (50a-c) we can get the values::

                    from arc import *

                    channels = [[70,0,0.5, 70, 1,1.5, 69,1, 1.5],\\
                                [70,0,0.5, 70, 1,1.5, 69,1, 0.5],\\
                                [70,0,0.5, 69, 1,1.5, 70,1, 0.5],\\
                                [70,0,0.5, 70, 1,0.5, 69,1, 0.5]]

                    print(" = = = Caesium = = = ")
                    atom = Caesium()
                    for channel in channels:
                        print("%.0f  GHz (mu m)^6" % ( atom.getC6term(*channel) / C_h * 1.e27 ))

                    print("\\n = = = Rubidium  = = =")
                    atom = Rubidium()
                    for channel in channels:
                        print("%.0f  GHz (mu m)^6" % ( atom.getC6term(*channel) / C_h * 1.e27 ))

                Returns::

                     = = = Caesium = = =
                    722  GHz (mu m)^6
                    316  GHz (mu m)^6
                    383  GHz (mu m)^6
                    228  GHz (mu m)^6

                     = = = Rubidium  = = =
                    799  GHz (mu m)^6
                    543  GHz (mu m)^6
                    589  GHz (mu m)^6
                    437  GHz (mu m)^6

                which is in good agreement with the values cited in the Ref. [#c6r1]_.
                Small discrepancies for Caesium originate from slightly different
                quantum defects used in calculations.


            References:
                .. [#c6r1] T. G. Walker, M. Saffman, PRA **77**, 032723 (2008)
                    https://doi.org/10.1103/PhysRevA.77.032723

        """
        d1 = self.getDipoleMatrixElement(n,l,j,m,n1,l1,j1,m1,q1,semi)
        d2 = self.getDipoleMatrixElement(n,l,j,mm,n2,l2,j2,m2,q2,semi)

        #print(d1,self.getRadialMatrixElementSemiClassical(n,l,j,n1,l1,j1))
        #print(d2 , self.getRadialMatrixElementSemiClassical(n,l,j,n2,l2,j2))

        d1d2 = 1/(4.0*pi*epsilon_0)*d1*d2*C_e**2*\
                (physical_constants["Bohr radius"][0])**2
        #d1d2 = d1*d2

        return -d1d2**2/(C_e*(self.getEnergy(n1,l1,j1)+\
                                     self.getEnergy(n2,l2,j2)-\
                                     2*self.getEnergy(n,l,j)))          

    def getC3term(self,n,l,j,n1,l1,j1,n2,l2,j2):
        """
            C3 interaction term for the given two pair-states

            Calculates :math:`C_3` intaraction term for :math:`|n,l,j,n,l,j\\rangle \
                 \\leftrightarrow |n_1,l_1,j_1,n_2,l_2,j_2\\rangle`

            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  :math:`C_3 = \\frac{\\langle n,l,j |er|n_1,l_1,j_1\\rangle \
                    \\langle n,l,j |er|n_2,l_2,j_2\\rangle}{4\\pi\\varepsilon_0}`
                (:math:`h` Hz m :math:`{}^3`).
        """
        d1 = self.getRadialMatrixElement(n,l,j,n1,l1,j1)
        d2 = self.getRadialMatrixElement(n,l,j,n2,l2,j2)
        d1d2 = 1/(4.0*pi*epsilon_0)*d1*d2*C_e**2*\
                (physical_constants["Bohr radius"][0])**2
        return d1d2     
        
    def getEnergyDefect(self,n,l,j,n1,l1,j1,n2,l2,j2,s = 0.5):
        """
            Energy defect for the given two pair-states (one of the state has
            two atoms in the same state)

            Energy difference between the states
            :math:`E(n_1,l_1,j_1,n_2,l_2,j_2) - E(n,l,j,n,l,j)`

            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  energy defect (SI units: J)
        """
        return C_e*(self.getEnergy(n1,l1,j1,s)+self.getEnergy(n2,l2,j2,s)-\
                           2*self.getEnergy(n,l,j,s))

    def getEnergyDefect2(self,n,l,j,nn,ll,jj,n1,l1,j1,n2,l2,j2,s=0.5):
        """
            Energy defect for the given two pair-states

            Energy difference between the states
            :math:`E(n_1,l_1,j_1,n_2,l_2,j_2) - E(n,l,j,nn,ll,jj)`

            See `pair-state energy defects example snippet`_.

            .. _`pair-state energy defects example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-atom-interactions


            Args:
                n (int): principal quantum number
                l (int): orbital angular momenutum
                j (float): total angular momentum
                nn (int): principal quantum number
                ll (int): orbital angular momenutum
                jj (float): total angular momentum
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  energy defect (SI units: J)
        """
        return (self.getEnergy(n1,l1,j1,s)+self.getEnergy(n2,l2,j2,s)-\
                           self.getEnergy(n,l,j,s)-self.getEnergy(nn,ll,jj,s))*C_e

    def updateDipoleMatrixElementsFile(self):
        """
            Updates the file with pre-calculated dipole matrix elements.

            This function will add the the file all the elements that have been
            calculated in the previous run, allowing quick access to them in the
            future calculations.
        """
        # obtain dipole matrix elements from the database

        dipoleMatrixElement = []
        self.c.execute('''SELECT * FROM dipoleME ''')
        for v in self.c.fetchall():
            dipoleMatrixElement.append(v)

        # obtain quadrupole matrix elements from the database

        quadrupoleMatrixElement = []
        self.c.execute('''SELECT * FROM quadrupoleME ''')
        for v in self.c.fetchall():
            quadrupoleMatrixElement.append(v)

        # save dipole elements
        try:
            np.save(os.path.join(self.dataFolder,\
                                 self.dipoleMatrixElementFile),\
                    dipoleMatrixElement)
        except IOError as e:
            print("Error while updating dipoleMatrixElements File "+\
                    self.dipoleMatrixElementFile)
            print(e)
        # save quadrupole elements
        try:
            np.save(os.path.join(self.dataFolder,\
                                 self.quadrupoleMatrixElementFile),\
                    quadrupoleMatrixElement)
        except IOError as e:
            print("Error while updating quadrupoleMatrixElements File "+\
                    self.quadrupoleMatrixElementFile)
            print(e)

    def getRadialCoupling(self,n,l,j,n1,l1,j1,s=0.5):
        """
            Returns radial part of the coupling between two states (dipole and
            quadrupole interactions only)

            Args:
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum

            Returns:
                float:  radial coupling strength (in a.u.), or zero for forbidden
                transitions in dipole and quadrupole approximation.

        """
        dl = abs(l-l1)

        # LIZZY We currently don't have the radial model potetnial working yet
        #so if it is strontium then we have to use the semiClassical

        if (dl == 1 and abs(j-j1)<1.1):
            #print(n," ",l," ",j," ",n1," ",l1," ",j1)
            return self.getRadialMatrixElement(n,l,j,n1,l1,j1,s)
           
        elif (dl==0 or dl==1 or dl==2) and(abs(j-j1)<2.1):
            # quadrupole coupling
            #return 0.
            return self.getQuadrupoleMatrixElement(n,l,j,n1,l1,j1,s)
        else:
            # neglect octopole coupling and higher
            #print("NOTE: Neglecting couplings higher then quadrupole")
            return 0

    def getAverageSpeed(self,temperature):
        """
            Average (mean) speed at a given temperature

            Args:
                temperature (float): temperature (K)

            Returns:
                float: mean speed (m/s)
        """
        return sqrt( 8.*C_k*temperature/(pi*self.mass) )

    def _readLiteratureValues(self):
        # clear previously saved results, since literature file
        # might have been updated in the meantime
        
        #Lizzy GOING TO HAVE TO SORT THIS OUT PROPERLY
        
        self.c.execute('''DROP TABLE IF EXISTS literatureDME''')
        self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                        WHERE type='table' AND name='literatureDME';''')
        if (self.c.fetchone()[0] == 0):
            # create table
            self.c.execute('''CREATE TABLE IF NOT EXISTS literatureDME
             (n1 TINYINT UNSIGNED, l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
             n2 TINYINT UNSIGNED, l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
             dme DOUBLE,
             s TINYINT UNSIGNED,
             typeOfSource TINYINT,
             errorEstimate DOUBLE,
             comment TINYTEXT,
             ref TINYTEXT,
             refdoi TINYTEXT
            );''')
            self.c.execute('''CREATE INDEX compositeIndex
            ON literatureDME (n1,l1,j1_x2,n2,l2,j2_x2,s); ''')
        self.conn.commit()

        if (self.literatureDMEfilename == ""):
            return 0; # no file specified for literature values

        try:
            fn = open(os.path.join(self.dataFolder,self.literatureDMEfilename), 'r')
            data= csv.reader(fn,delimiter=",",quotechar='"')

            literatureDME = []

            # i=0 is header
            i=0
            for row in data:
                if i!=0:
                    n1 = int(row[0])
                    l1 = int(row[1])
                    j1 = float(row[2])
                    n2 = int(row[3])
                    l2 = int(row[4])
                    j2 = float(row[5])
                    s = float(row[6])/2
                    if (self.getEnergy(n1, l1, j1, s)>self.getEnergy(n2, l2, j2,s)):
                        temp = n1
                        n1 = n2
                        n2 = temp
                        temp = l1
                        l1 = l2
                        l2 = temp
                        temp = j1
                        j1 = j2
                        j2 = temp

                    # convered from reduced DME in J basis (symmetric notation)
                    # to radial part of dme as it is saved for calculated values
                    dme = float(row[7])/((-1)**(int(l1+0.5+j2+1.))*\
                                sqrt((2.*j1+1.)*(2.*j2+1.))*\
                                Wigner6j(j1, 1., j2, l2, 0.5, l1)*\
                                (-1)**l1*sqrt((2.0*l1+1.0)*(2.0*l2+1.0))*\
                                Wigner3j(l1,1,l2,0,0,0))

                    comment = row[8]
                    if(len(row) == 13):
                        typeOfSource = int(row[9])  # 0 = experiment; 1 = theory
                        errorEstimate = float(row[10])
                        ref = row[11]
                        refdoi = row[12]
                        
                    else:
                        

                        typeOfSource = int(row[10])  # 0 = experiment; 1 = theory
                        errorEstimate = float(row[11])
                        ref = row[12]
                        refdoi = row[13]

                    literatureDME.append([n1,l1,j1*2,n2,l2,j2*2,int(s*2),dme,\
                                               typeOfSource,errorEstimate,\
                                               comment,ref,\
                                                    refdoi])
                i +=1
            fn.close()

            try:
                self.c.executemany('''INSERT INTO literatureDME
                                    VALUES (?,?,?,?,?,?,?,
                                            ?,?,?,?,?,?)''',\
                                     literatureDME)
                self.conn.commit()
            except sqlite3.Error as e:
                print("Error while loading precalculated values into the database")
                print(e)
                exit()




        except IOError as e:
            print("Error reading literature values File "+\
                    self.literatureDMEfilename)
            print(e)



    def getLiteratureDME(self,n1,l1,j1,n2,l2,j2,s =0.5):
        """
            Returns literature information on requested transition.

            Args:
                n1,l1,j1: one of the states we are coupling
                n2,l2,j2: the other state to which we are coupling

            Returns:
                bool, float, [int,float,string,string,string]:

                    hasLiteratureValue?, dme, referenceInformation

                    **If Boolean value is True**, a literature value for dipole matrix
                    element was found and reduced DME in J basis is returned
                    as the number. Third returned argument (array) contains
                    additional information about the literature value in the
                    following order [ typeOfSource, errorEstimate , comment ,Fge
                    reference, reference DOI] upon success to
                    find a literature value for dipole matrix element:
                        * typeOfSource=1 if the value is theoretical calculation;\
                         otherwise, if it is experimentally obtained value\
                         typeOfSource=0
                        * comment details where within the publication the value\
                         can be found
                        * errorEstimate is absolute error estimate
                        * reference is human-readable formatted reference
                        * reference DOI provides link to the publication.

                    **Boolean value is False**, followed by zero and an empty array
                    if no literature value for dipole matrix element is found.

            Note:
                The literature values are stored in /data folder in
                <element name>_literature_dme.csv files as a ; separated values.
                Each row in the file consists of one literature entry, that has
                information in the following order:

                 * n1
                 * l1
                 * j1
                 * n2
                 * l2
                 * j2
                 * dipole matrix element reduced l basis (a.u.)
                 * comment (e.g. where in the paper value appears?)
                 * value origin: 1 for theoretical; 0 for experimental values
                 * accuracy
                 * source (human readable formatted citation)
                 * doi number (e.g. 10.1103/RevModPhys.82.2313 )

                If there are several values for a given transition, program will
                output the value that has smallest error (under column accuracy).
                The list of values can be expanded - every time program runs
                this file is read and the list is parsed again for use in
                calculations.

        """

        if (self.getEnergy(n1, l1, j1,s)>self.getEnergy(n2, l2, j2,s)):
            temp = n1
            n1 = n2
            n2 = temp
            temp = l1
            l1 = l2
            l2 = temp
            temp = j1
            j1 = j2
            j2 = temp


        # is there literature value for this DME? If there is,
        # use the best one (wit the smallest error)

        j1_x2 = int(round(2*j1))
        j2_x2 = int(round(2*j2))
        s = int(s)

        self.c.execute('''SELECT dme, typeOfSource,
                     errorEstimate ,
                     comment ,
                     ref,
                     refdoi FROM literatureDME WHERE
                     n1= ? AND l1 = ? AND j1_x2 = ? AND
                     n2 = ? AND l2 = ? AND j2_x2 = ? AND s =?
                     ORDER BY errorEstimate ASC''',\
                     (n1,l1,j1_x2,n2,l2,j2_x2,s))
        answer = self.c.fetchone()
        if (answer):
            # we did found literature value
            return True,answer[0],[answer[1],answer[2],answer[3],\
                                   answer[4],answer[5]]

        # if we are here, we were unsucessfull in literature search for this value
        return False,0,[]

    def getZeemanEnergyShift(self, l, j, mj, magneticFieldBz):
        r"""
            Retuns linear (paramagnetic) Zeeman shift.

            :math:`\mathcal{H}_P=\frac{\mu_B B_z}{\hbar}(\hat{L}_{\rm z}+g_{\rm S}S_{\rm z})`

            Returns:
                float: energy offset of the state (in J)
        """
        prefactor = physical_constants["Bohr magneton"][0] * magneticFieldBz
        gs = - physical_constants["electron g factor"][0]
        sumOverMl = 0
        if (mj+0.5 < l + 0.1):
            # include ml = mj + 1/2
            sumOverMl = (mj + 0.5 - gs * 0.5) * \
                        abs(CG(l, mj + 0.5, 0.5, -0.5, j, mj))**2
        if (mj -0.5 > -l - 0.1):
            # include ml = mj - 1/2
            sumOverMl += (mj - 0.5 + gs * 0.5) * \
                         abs(CG(l, mj - 0.5, 0.5, 0.5, j, mj))**2
        return prefactor * sumOverMl

    
    #==================== Quantum Defect fitting routine =============
    #This code has been developed by Paul Huillery.
    def QuantumDefectFit(self,filename):
        '''filename- str  file name for quantum defects to be written to
        
            filename will have defects written to them in form:
                Series, lowern, uppern, \delta0, delta2, delta 4, energy_ref
        '''
    
        def E(n,d0,d2,d4) :
            return self.ionisationEnergycm - self.scaledRydbergConstant/(n-(d0 + d2/(n-d0)**2 + d4/(n-d0)**4))**2
    
        def fitting(E_data, n_data, nLower, nUpper, series ):
            '''
            This will fit the quantum defects for a sub set of n values. These will then have to be averaged over
            to obtain the quantum defect 
            
            '''
            #if there is that range 
            if(nLower in n_data) and (nUpper in n_data):
            
                #we need to select the data
                lower_idx = int(np.where(n_data == nLower)[0])
                upper_idx = int(np.where(n_data == nUpper)[0])
                
                #print(lower_idx)
                #print(upper_idx)
                
                E_data = E_data[lower_idx: upper_idx+1]
                n_data = n_data[lower_idx: upper_idx+1]
                
                #print(E_data)
                #print(n_data)
                
                E_err = np.zeros(len(n_data))+0.001
                
                indexes = np.arange(len(n_data))
                 
                nb_of_type = 5
                nb_of_subset = 1000
                my_subsets = []
                 
                for t in range(nb_of_type) :
                    for i in range(nb_of_subset) :
                        cutted_indexes = np.random.choice(indexes,15+t)
                        
                        #cutted_indexes = indexes
                        my_subsets.append(cutted_indexes)
                 
                res = np.zeros((nb_of_type*nb_of_subset,3))
                chi_2 = np.zeros(nb_of_type*nb_of_subset)
                 
                initial_guess = (2,0,0)
                 
                a = 100
                
                for i in range(nb_of_type*nb_of_subset) :
             
                    this_n_data, this_qd_data, this_err_data = [], [], []
                    
                    for d in range(len(my_subsets[i])) :
                        this_n_data.append(n_data[my_subsets[i][d]])
                        if n_data[my_subsets[i][d]] < a :
                           a = n_data[my_subsets[i][d]]
                        this_qd_data.append(E_data[my_subsets[i][d]])
                        this_err_data.append(E_err[my_subsets[i][d]])
                        
                    #turn to numpy arrays     
                    this_n_data = np.array(this_n_data)
                    this_qd_data = np.array(this_qd_data)
                    this_err_data = np.array(this_err_data)
                    #if series== '1D2':
                    #    print(this_n_data)
                    #    print(this_qd_data)
                    #    print(this_err_data)
                    
                    #do the fitting
                    popt_s, pcov_s = curve_fit(E, this_n_data, this_qd_data, initial_guess, this_err_data, True)
                    res[i,0] = popt_s[0]
                    res[i,1] = popt_s[1]
                    res[i,2] = popt_s[2]
                     
                    #print pcov_s[0,0]**0.5
                 
                    for j in range(len(my_subsets[i])) :
                        delta = E(n_data[my_subsets[i][j]],popt_s[0],popt_s[1],popt_s[2]) - E_data[my_subsets[i][j]]
                        sigma = E_err[my_subsets[i][j]]
                        chi_2[i] += (delta/sigma)**2
                         
                    chi_2[i] /= len(my_subsets[i])-3   
                
                # we take the quantum defects to be the mean of all quantum defects fitted. 
                return res, chi_2
            else:
                print('There is no data entry for either '+ str(nLower) + ' or ' +str(nUpper))
                print('Please select a different range' )
            return [0],0 
        
        csv_data = []
        
        
        #this will refit the quantum defects for all energy files given in self.NISTDATAFILES
        
        for file in self.levelDataFromNIST:
            #get the series from the filename
            series = str(file[3:6])
            print(series)
            
            data = np.genfromtxt(DPATH+'/'+file, delimiter = ',')
            
            
            mask = (~np.isnan(data[:,0:2])).all(axis =1)
            
            data = data[mask,:]
            #print(data)
                
            n = data[:,0]
            e = data[:,1]
            
            #remove nan. 
            
            
            #print(self.defectFittingRange)
            lower_fit_n = self.defectFittingRange[series][0]
            upper_fit_n = self.defectFittingRange[series][1]
            
            
            qd_coeffs, chi_squared = fitting(e,n, lower_fit_n, upper_fit_n,series)
            
            print('chi_squared',np.mean(chi_squared))
            #if (series == '1D2'):
            #    print(e)
            
            #S    print(qd_coeffs)
            if (len(qd_coeffs) != 1):
            #w
                deltas = []
                deltas.append(np.mean(qd_coeffs[:,0])) #np.std(qd_coeffs[:,0]))
                deltas.append(np.mean(qd_coeffs[:,1])) #np.std(qd_coeffs[:,1]))
                deltas.append(np.mean(qd_coeffs[:,2])) #np.std(qd_coeffs[:,2]))
                

                print(np.mean(qd_coeffs[:,0]),np.std(qd_coeffs[:,0]))#
                print(np.mean(qd_coeffs[:,1]),np.std(qd_coeffs[:,1]))
                print(np.mean(qd_coeffs[:,2]) ,np.std(qd_coeffs[:,2]))
                csv_data.append([series,lower_fit_n,upper_fit_n,deltas[0],deltas[1],deltas[2],'This work'])
            else:
                csv_data.append([series,lower_fit_n,upper_fit_n,'fit','not','possible','This work'])

        #print(np.array(csv_data))
        np.savetxt(filename,np.array(csv_data),delimiter = "," , fmt='%s')
            
        return 
        


def NumerovBack(innerLimit,outerLimit,kfun,step,init1,init2):
    """
        Full Python implementation of Numerov integration

        Calculates solution function :math:`rad(r)` with descrete step in
        :math:`r` size of `step`, integrating from `outerLimit` towards the
        `innerLimit` (from outside, inwards) equation
        :math:`\\frac{\\mathrm{d}^2 rad(r)}{\\mathrm{d} r^2} = kfun(r)\\cdot rad(r)`.




        Args:
            innerLimit (float): inner limit of integration
            outerLimit (flaot): outer limit of integration
            kfun (function(double)): pointer to function used in equation (see
                longer explanation above)
            step: descrete step size for integration
            init1 (float): initial value, `rad`(`outerLimit`+`step`)
            init2 (float): initial value, `rad`(`outerLimit`+:math:`2\\cdot` `step`)

        Returns:
            numpy array of float , numpy array of float, int : :math:`r` (a.u),
            :math:`rad(r)`;

        Note:
            Returned function is not normalized!

        Note:
            If :obj:`AlkaliAtom.cpp_numerov` swich is set to True (default option),
            much faster C implementation of the algorithm will be used instead.
            That is recommended option. See documentation installation
            instructions for more details.

    """

    br = int((sqrt(outerLimit)-sqrt(innerLimit))/step)
    sol = np.zeros(br,dtype=np.dtype('d'))  # integrated wavefunction R(r)*r^{3/4}
    rad = np.zeros(br,dtype=np.dtype('d'))  # radial coordinate for integration \sqrt(r)

    br = br-1
    x = sqrt(innerLimit)+step*(br-1)
   
    sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*init1-\
               (1.+1./12.0*step**2*kfun(x+step))*init2)/\
               (1+1/12.0*step**2*kfun(x-step))
    rad[br] = x

    
    x = x-step
    br = br-1

    sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*sol[br+1]-\
               (1.+1./12.0*step**2*kfun(x+step))*init1)/\
               (1+1/12.0*step**2*kfun(x-step))
    rad[br] = x

    # check if the function starts diverging  before the innerLimit
    # -> in that case break integration earlier

    maxValue = 0.

    checkPoint = 0
    fromLastMax = 0

    while br>checkPoint:
        br = br-1
        x = x-step
        sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*sol[br+1]-\
                   (1.+1./12.0*step**2*kfun(x+step))*sol[br+2])/\
                   (1.+1./12.0*step**2*kfun(x-step))
        rad[br] = x
        if abs(sol[br]*sqrt(x))>maxValue:
            maxValue = abs(sol[br]*sqrt(x))
        else:
            fromLastMax += 1
            if fromLastMax>50:
                checkPoint = br
    # now proceed with caution - checking if the divergence starts
    # - if it does, cut earlier

    divergencePoint = 0

    while (br>0)and(divergencePoint==0):
        br = br-1
        x = x-step
        sol[br] = (2.*(1.-5.0/12.0*step**2*kfun(x))*sol[br+1]-\
                   (1.+1./12.0*step**2*kfun(x+step))*sol[br+2])/\
                   (1.+1./12.0*step**2*kfun(x-step))
        rad[br] = x
        if (divergencePoint==0)and(abs(sol[br]*sqrt(x))>maxValue):
            divergencePoint = br
            while( abs(sol[divergencePoint])>abs(sol[divergencePoint+1])) and \
                (divergencePoint<checkPoint):
                divergencePoint +=1
            if divergencePoint>checkPoint:
                print("Numerov error")
                exit()

    br = divergencePoint;
    while (br>0):
        rad[br]=rad[br+1]-step;
        sol[br]=0;
        br -= 1;

    # convert R(r)*r^{3/4} to  R(r)*r
    sol = np.multiply(sol,np.sqrt(rad))
    # convert \sqrt(r) to r
    rad = np.multiply(rad,rad)

    return rad,sol


def _atomLightAtomCoupling(n,l,j,nn,ll,jj,n1,l1,j1,n2,l2,j2,atom,s=0.5):
    """
        Calculates radial part of atom-light coupling

        This function might seem redundant, since similar function exist for
        each of the atoms. However, function that is not connected to specific
        atomic species is provided in order to provides route to implement
        inter-species coupling in the future.
    """
    # determine coupling
    dl = abs(l-l1)
    dj = abs(j-j1)
    c1 = 0
    if dl==1 and (dj<1.1):
        c1 = 1  # dipole coupling
    elif (dl==0 or dl==2 or dl==1) and(dj<2.1):
        c1 = 2  # quadrupole coupling
    else:
        return False
    dl = abs(ll-l2)
    dj = abs(jj-j2)
    c2 = 0
    if dl==1 and (dj<1.1):
        c2 = 1  # dipole coupling
    elif (dl==0 or dl==2 or dl==1) and(dj<2.1):
        c2 = 2  # quadrupole coupling
    else:
        return False

    radial1 = atom.getRadialCoupling(n,l,j,n1,l1,j1,s)
    radial2 = atom.getRadialCoupling(nn,ll,jj,n2,l2,j2,s)
    ## TO-DO: check exponent of the Boht radius (from where it comes?!)
    #print('radial1', radial1)
    #print('radial2', radial2)
    coupling = C_e**2/(4.0*pi*epsilon_0)*radial1*radial2*\
                (physical_constants["Bohr radius"][0])**(c1+c2)
    #coupling= radial1*radial2
    return coupling


# =================== Saving and loading calculations (START) ===================

def saveCalculation(calculation,fileName):
    """
    Saves calculation for future use.

    Saves :obj:`calculations_atom_pairstate.PairStateInteractions` and
    :obj:`calculations_atom_single.StarkMap`
    calculations in compact binary format in file named `filename`. It uses
    cPickle serialization library in Python, and also zips the final file.

    Calculation can be retrieved and used with :obj:`loadSavedCalculation`

    Args:
        calculation: class instance of calculations (instance of
            :obj:`calculations_atom_pairstate.PairStateInteractions`c in xcoords:
    plt.axvline(x=xc)

            or :obj:`calculations_atom_single.StarkMap`)
            to be saved.
        fileName: name of the file where calculation will be saved

    Example:
        Let's suppose that we did the part of the
        :obj:`calculation_atom_pairstate.PairStateInteractions`
        calculation that involves generation of the interaction
        matrix. After that we can save the full calculation in a single file::

            calc = PairStateInteractions(Rubidium(), 60,0,0.5,60,0,0.5, 0.5,0.5)
            calc.defineBasis(0,0, 5,5, 25.e9)
            calc.diagonalise(np.linspace(0.5,10.0,200),150)
            saveCalculation(calc, "mySavedCalculation.pkl")

        Then, at a later time, and even on the another machine, we can load
        that file and continue with calculation. We can for example explore
        the calculated level diagram::

            calc = loadSavedCalculation("mySavedCalculation.pkl")
            calc.plotLevelDiagram()
            calc.showPlot()
            rvdw = calc.getVdwFromLevelDiagram(0.5,14,minStateContribution=0.5,\\
                                               showPlot = True)

        Or, we can do additional matrix diagonalization, in some new range,
        then and find C6 by fitting the obtained level diagram::

            calc = loadSavedCalculation("mySavedCalculation.pkl")
            calc.diagonalise(np.linspace(3,6.0,200),20)
            calc.getC6fromLevelDiagram(3,6.0,showPlot=True)

        Note that for all loading of saved calculations we've been using
        function :obj:`loadSavedCalculation` .


    Note:
        This doesn't save results of :obj:`plotLevelDiagram` for the corresponding
        calculations. Call the plot function before calling :obj:`showPlot` function
        for the corresponding calculation.

    """

    try:
        ax  = calculation.ax
        fig = calculation.fig
        calculation.ax = 0
        calculation.fig = 0

        # close database connections
        atomDatabaseConn = calculation.atom.conn
        atomDatabaseC = calculation.atom.c
        calculation.atom.conn = False
        calculation.atom.c = False

        output = gzip.GzipFile(fileName, 'wb')
        pickle.dump(calculation, output, pickle.HIGHEST_PROTOCOL)
        output.close()

        calculation.ax = ax
        calculation.fig = fig
        calculation.atom.conn = atomDatabaseConn
        calculation.atom.c = atomDatabaseC
    except:
        print("ERROR: saving of the calculation failed.")
        print(sys.exc_info())
        return 1
    return 0

def loadSavedCalculation(fileName):
    """
    Loads previously saved calculation.

    Loads :obj:`calculations_atom_pairstate.PairStateInteractions` and
    :obj:`calculations_atom_single.StarkMap`
    calculation instance from file named `filename` where it was previously saved
    with :obj:`saveCalculation` .

    Example:
        See example for :obj:`saveCalculation`.

    Args:
        fileName: name of the file where calculation will be saved

    Returns:
        saved calculation
    """

    calculation = False
    try:
        calcInput = gzip.GzipFile(fileName, 'rb')
        calculation = pickle.load(calcInput)
    except:
        print("ERROR: loading of the calculation from '%s' failed"  % fileName)
        print(sys.exc_info())
        return False
    print("Loading of "+calculation.__class__.__name__+" from '"+fileName+\
        "' successful.")

    # establish conneciton to the database
    calculation.atom._databaseInit()

    return calculation

# =================== Saving and loading calculations (END) ===================

# =================== State generation and printing (START) ===================

def singleAtomState(j,m):
    a = zeros((int(round(2.0*j+1.0,0)),1),dtype=np.complex128)
    a[int(round(j+m,0))] = 1
    return a
    return csr_matrix(([1], ([j+m], [0])),
                                       shape=(round(2.0*j+1.0,0),1))

def compositeState(s1,s2):
    a = zeros((s1.shape[0]*s2.shape[0],1),dtype=np.complex128)
    index = 0
    for br1 in xrange(s1.shape[0]):
        for br2 in xrange(s2.shape[0]):
            a[index] = s1[br1]*s2[br2]
            index += 1
    return a

def printState(n,l,j,s=0.5):
    """
        Prints state spectroscopic label for numeric :math:`n`,
        :math:`l`, :math:`s` label of the state

        Args:
            n (int): principal quantum number
            l (int): orbital angular momentum
            j (float): total angular momentum
    """
    if(j % 1 !=0 ):
        print( str(n)+" "+printStateLetter(l)+(" %.0d/2" % (j*2)))
    else:
        print( str(n)+" "+(" %.0d " % (2*s+1))+printStateLetter(l)+(" %.0d" % (j)))
    return
def printStateString(n,l,j,s=0.5):
    """
        Returns state spectroscopic label for numeric :math:`n`,
        :math:`l`, :math:`s` label of the state

        Args:
            n (int): principal quantum number
            l (int): orbital angular momentum
            j (float): total angular momentum

        Returns:
            string: label for the state in standard spectroscopic notation
    """
    if(j % 1 !=0 ):
        return str(n)+" "+printStateLetter(l)+(" %.0d/2" % (j*2))
    else:
        return str(n)+" "+(" %.0d " % (2*s+1))+printStateLetter(l)+(" %.0d" % (j))
def printStateStringLatex(n,l,j,s=0.5):
    """
        Returns latex code for spectroscopic label for numeric :math:`n`,
        :math:`l`, :math:`s` label of the state

        Args:
            n (int): principal quantum number
            l (int): orbital angular momentum
            j (float): total angular momentum

        Returns:
            string: label for the state in standard spectroscopic notation
    """
    if(j % 1 !=0 ):
        return str(n)+printStateLetter(l)+("_{%.0d/2}" % (j*2))
    else:
        return str(n)+" "+(" ^{%.0d}" % (2*s+1))+printStateLetter(l)+("_{%.0d}" % (j))

def printStateLetter(l):
    let = ''
    if l==0:
        let = "S"
    elif l==1:
        let = "P"
    elif l == 2:
        let = "D"
    elif l== 3:
        let = "F"
    elif l == 4:
        let = "G"
    elif l == 5:
        let = "H"
    elif l == 6:
        let = "I"
    elif l == 7:
        let = "K"
    elif l == 8:
        let = "L"
    elif l == 9:
        let = "M"
    elif l == 10:
        let = "N"
    else:
        let = " l=%d" % l
    return let

# =================== State generation and printing (END) ===================

# =================== E FIELD Coupling (START) ===================

class _EFieldCoupling:
    dataFolder = DPATH

    def __init__(self, theta=0., phi=0.,s =0.5):
        self.theta = theta
        self.phi = phi
        self.s =s

        # STARK memoization
        self.conn = sqlite3.connect(os.path.join(self.dataFolder,\
                                                 "precalculated_stark.db"))
        self.c = self.conn.cursor()


        ### ANGULAR PARTS

        self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='eFieldCoupling_angular';''')
        if (self.c.fetchone()[0] == 0):
            # create table
            self.c.execute('''CREATE TABLE IF NOT EXISTS eFieldCoupling_angular
             (l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED, j1_mj1 TINYINT UNSIGNED,
              l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED, j2_mj2 TINYINT UNSIGNED,
             sumPart DOUBLE,s TINYINT UNSIGNED,
             PRIMARY KEY (l1,j1_x2,j1_mj1,l2,j2_x2,j2_mj2,s)
            ) ''')
            self.conn.commit()

        ###COUPLINGS IN ROTATED BASIS (depend on theta, phi)
        self.wgd = wignerDmatrix(self.theta, self.phi)

        self.c.execute('''DROP TABLE IF EXISTS eFieldCoupling''')
        self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='eFieldCoupling';''')
        if (self.c.fetchone()[0] == 0):
            # create table
            self.c.execute('''CREATE TABLE IF NOT EXISTS eFieldCoupling
             (l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED, j1_mj1 TINYINT UNSIGNED,
              l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED, j2_mj2 TINYINT UNSIGNED,
             coupling DOUBLE, s TINYINT UNSIGNED,
             PRIMARY KEY (l1,j1_x2,j1_mj1,l2,j2_x2,j2_mj2,s)
            ) ''')
            self.conn.commit()

    def getAngular(self,l1,j1,mj1,l2,j2,mj2,s = 0.5):
        #self.c.execute('''SELECT sumPart FROM eFieldCoupling_angular WHERE
        # l1= ? AND j1_x2 = ? AND j1_mj1 = ? AND
        # l2 = ? AND j2_x2 = ? AND j2_mj2 = ? and s = ?
        # ''',(l1,2*j1,j1+mj1,l2,j2*2,j2+mj2,s*2))
        answer = self.c.fetchone()
        if (answer):
            return answer[0]

        # calulates sum (See PRA 20:2251 (1979), eq.(10))
        sumPart = 0.
        ml = mj1 + 0.5
        if (abs(ml)-0.1<l1)and(abs(ml)-0.1<l2):

            angularPart = 0.
            if (abs(l1-l2-1)<0.1):
                angularPart = ((l1**2-ml**2)/((2.*l1+1.)*(2.*l1-1.)))**0.5
            elif(abs(l1-l2+1)<0.1):
                angularPart = ((l2**2-ml**2)/((2.*l2+1.)*(2.*l2-1.)))**0.5

            sumPart += CG(l1,ml,0.5,mj1-ml,j1,mj1)*CG(l2,ml,0.5,mj1-ml,j2,mj2)*\
                        angularPart


        ml = mj1 - 0.5
        if (abs(ml)-0.1<l1)and(abs(ml)-0.1<l2):
            angularPart = 0.
            if (abs(l1-l2-1)<0.1):
                angularPart = ((l1**2-ml**2)/((2.*l1+1.)*(2.*l1-1.)))**0.5
            elif(abs(l1-l2+1)<0.1):
                angularPart = ((l2**2-ml**2)/((2.*l2+1.)*(2.*l2-1.)))**0.5
            sumPart += CG(l1,ml,0.5,mj1-ml,j1,mj1)*CG(l2,ml,0.5,mj1-ml,j2,mj2)*\
                        angularPart

        #self.c.execute(''' INSERT INTO eFieldCoupling_angular
        #                    VALUES (?,?,?, ?,?,?,?, ?)''',\
        #                    [l1,2*j1,j1+mj1,l2,j2*2,j2+mj2,s*2,sumPart] )
        #self.conn.commit()

        return sumPart

    def getCouplingDivEDivDME(self,l1,j1,mj1,l2,j2,mj2, s=0.5):
        # returns angular coupling without radial part and electric field

        # if calculated before, retrieve from memory
        self.c.execute('''SELECT coupling FROM eFieldCoupling WHERE
         l1= ? AND j1_x2 = ? AND j1_mj1 = ? AND
         l2 = ? AND j2_x2 = ? AND j2_mj2 = ?, s = ?
         ''',(l1,2*j1,j1+mj1,l2,j2*2,j2+mj2,s))
        answer = self.c.fetchone()
        if (answer):
            return answer[0]

        # if it is not calculated before, calculate now

        coupling = 0.

        ## rotate individual states
        statePart1 = singleAtomState(j1, mj1)
        dMatrix = self.wgd.get(j1)
        statePart1 = np.conj(dMatrix.dot(statePart1))

        statePart2 = singleAtomState(j2, mj2)
        dMatrix = self.wgd.get(j2)
        statePart2 = dMatrix.dot(statePart2)

        ## find first common index and start summation
        start = min(j1,j2)

        for mj in np.linspace(-start,start,floor(2*start+1)):
            coupling += (self.getAngular(l1,j1,mj,l2,j2,mj)*\
                        (statePart1[j1+mj]*statePart2[j2+mj])[0].real)

        ## save in memory for later use

        self.c.execute(''' INSERT INTO eFieldCoupling
                            VALUES (?,?,?, ?,?,?, ?,?)''',\
                            [l1,2*j1,j1+mj1,l2,j2*2,j2+mj2,coupling,s] )
        self.conn.commit()

        # return result

        return coupling

    def _closeDatabase(self):
        self.conn.commit()
        self.conn.close()
        self.conn = False
        self.c = False

# =================== E FIELD Coupling (END) ===================

# we copy the data files to the user home at first run. This avoids
# permission trouble.


setup_data_folder()

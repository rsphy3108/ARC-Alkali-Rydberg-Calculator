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
from .general_atom_functions import Atom
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


class AlkaliAtom(Atom):

    quantumDefect = [[[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],\
                      [0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],\
                      [0.0,0.0,0.0,0.0,0.0,0.0]],
                     [[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],\
                      [0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0],\
                      [0.0,0.0,0.0,0.0,0.0,0.0]]]
                      
    a1,a2,a3,a4,rc = [0],[0],[0],[0],[0]
    """
        Model potential parameters fitted from experimental observations for
        different l (electron angular momentum)
    """

    def __init__(self,preferQuantumDefects=True,cpp_numerov=True):

        # should the wavefunction be calculated with Numerov algorithm implemented in C; if false, it uses Python implementation that is much slower
        #This calls the Initalizer for the Atom function, to initalise 
        #all datebases and cursors
        super(AlkaliAtom,self).__init__(preferQuantumDefects,cpp_numerov)


        self.sEnergy = np.array([[0.0] * (self.NISTdataLevels+1)]
                                * (self.NISTdataLevels+1))


        # Always load NIST data on measured energy levels;
        # Even when user wants to use quantum defects, qunatum defects for
        # lowest lying state are not always so accurate, so below the
        # minQuantumDefectN cut-off (defined for each element separately)
        # getEnergy(...) will always return measured, not calculated energy levels
        if (self.levelDataFromNIST == ""):
            print("NIST level data file not specified. Only quantum defects will be used.")
        else:
            levels = self._parseLevelsFromNIST(os.path.join(self.dataFolder,\
                                               self.levelDataFromNIST))
            br = 0

            while br<len(levels):
                self._addEnergy(levels[br][0], levels[br][1],levels[br][2], levels[br][3])
                br = br+1

        # read Literature values for dipole matrix elements
        self._readLiteratureValues()

        return

    

    def corePotential(self,l,r):
        """ core potential felt by valence electron

            For more details about derivation of model potential see
            Ref. [#marinescu]_.

            Args:
                l (int): orbital angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: core potential felt by valence electron (in a.u. ???)

            References:

                .. [#marinescu] M. Marinescu, H. R. Sadeghpour, and A. Dalgarno
                    PRA **49**, 982 (1994), https://doi.org/10.1103/PhysRevA.49.982
        """

        return -self.effectiveCharge(l,r)/r-self.alphaC/(2*r**4)*(1-exp(-(r/self.rc[l])**6))

    def effectiveCharge(self,l,r):
        """ effective charge of the core felt by valence electron

            For more details about derivation of model potential see
            Ref. [#marinescu]_.

            Args:
                l (int): orbital angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: effective charge (in a.u.)
         """

        return 1.0+(self.Z-1)*exp(-self.a1[l]*r)-r*(self.a3[l]+self.a4[l]*r)*exp(-self.a2[l]*r)


    def potential(self,l,s,j,r):
        """ returns total potential that electron feels

            Total potential = core potential + Spin-Orbit interaction

            Args:
                l (int): orbital angular momentum
                s (float): spin angular momentum
                j (float): total angular momentum
                r (float): distance from the nucleus (in a.u.)
            Returns:
                float: potential (in a.u.)
        """
        if l<4:
            return self.corePotential(l,r)+self.alpha**2/(2.0*r**3)*(j*(j+1.0)-l*(l+1.0)-s*(s+1))/2.0
        else:
            # act as if it is a Hydrogen atom
            return -1./r+self.alpha**2/(2.0*r**3)*(j*(j+1.0)-l*(l+1.0)-s*(s+1))/2.0

    

    def _parseLevelsFromNIST(self,fileData):
        """
            Parses the level energies from file listing the NIST ASD data

            Args:
                fileData (str): path to the file containing NIST ASD data for the element
        """
        f = open(fileData,"r")
        l = 0
        n = 0
        levels = []
        for line in f:

            line = re.sub('[\[\]]', '', line)
            pattern = "\.\d*[spdfgh]"
            pattern2 = "\|\s+\d*/"
            pattern3 = "/\d* \|"
            pattern4 = "\| *\d*\.\d* *\|"
            match = re.search(pattern,line)
            if (match!= None):
                n = int(line[match.start()+1:match.end()-1])
            if (match!= None):
                ch = line[match.end()-1:match.end()]
                if ch == "s":
                    l=0
                elif ch =="p":
                    l = 1
                elif ch == "d":
                    l = 2
                elif ch == "f":
                    l = 3
                elif ch == "g":
                    l = 4
                elif ch == "h":
                    l = 5
                else:
                    print("Unidentified character in line:\n",line)
                    exit()

            match = re.search(pattern2,line)
            if (match != None):
                br1 = float(line[match.start()+2:match.end()-1])
                match = re.search(pattern3,line)
                br2 = float(line[match.start()+1:match.end()-2])
                match = re.search(pattern4,line)
                energyValue = float(line[match.start()+1:match.end()-1])
                levels.append([n,l,br1/br2,energyValue])
        f.close()
        return levels

    def _addEnergy(self,n,l,j,energyNIST):
        """
            Adding energy levels

            Accepts energy level relative to **ground state**, and
            saves energy levels, relative to the **ionization treshold**.

            Args:
                energyNIST (float): energy relative to the nonexcited level (= 0 eV)
        """
        #
        if abs(j-(l-0.5))<0.001:
            # j =l-1/2
            self.sEnergy[n, l] = energyNIST - self.ionisationEnergy
        else:
            # j = l+1/2
            self.sEnergy[l, n] = energyNIST - self.ionisationEnergy

    

    def getEnergy(self,n,l,j,s = 0.5):
        """
            Energy of the level relative to the ionisation level (in eV)

            Returned energies are with respect to the center of gravity of the
            hyperfine-split states.
            If `preferQuantumDefects` =False (set during initialization) program
            will try use NIST energy value, if such exists, falling back to energy
            calculation with quantum defects if the measured value doesn't exist.
            For `preferQuantumDefects` =True, program will always calculate
            energies from quantum defects (useful for comparing quantum defect
            calculations with measured energy level values).

            Args:
                n (int): principal quantum number
                l (int): orbital angular momentum
                j (float): total angular momentum

            Returns:
                float: state energy (eV)
        """
        if l>=n:
            raise ValueError("Requested energy for state l=%d >= n=%d !" % (l,n))

        if (isinstance(self,AlkaliAtom) and s != 0.5):
            raise ValueError('Spin state quantum number for AlkaliAtom must be 0.5 ')
        #if (isinstance(self,EarthAlkaliAtom) and s == 0.5):
        #    raise ValueError('Spin state for EarthAlkaliAtom mus be explicitly defined')

        if abs(j-(l-0.5))<0.001:
            # j = l-1/2
            # use NIST data ?
            if (not self.preferQuantumDefects or
                n<self.minQuantumDefectN)and(n <= self.NISTdataLevels) and \
                (abs(self.sEnergy[n,l])>1e-8):
                    return self.sEnergy[n,l]
            # else, use quantum defects
            defect = self.getQuantumDefect(n, l,j)
            return -self.scaledRydbergConstant/((n-defect)**2)

        elif abs(j-(l+0.5))<0.001:
            # j = l+1/2
            # use NIST data ?
            if (not self.preferQuantumDefects or
                n<self.minQuantumDefectN)and(n <= self.NISTdataLevels) and \
                (abs(self.sEnergy[l,n])>1e-8):
                    return self.sEnergy[l,n]

            # else, use quantum defects
            defect = self.getQuantumDefect(n, l,j)
            return -self.scaledRydbergConstant/((n-defect)**2)
        else:
            raise ValueError("j (=%.1f) is not equal to l+1/2 nor l-1/2 (l=%d)"%\
                             (j,l))



    def getQuantumDefect(self,n,l,j,s=0.5):
        """
            Quantum defect of the level.

            For an example, see `Rydberg energy levels example snippet`_.

            .. _`Rydberg energy levels example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-Atom-Energy-Levels

            Args:
                n (int): principal quantum number
                l (int): orbital angular momentum
                j (float): total angular momentum

            Returns:
                float: quantum defect
        """
        defect = 0.0
        if (l<5):
            if abs(j-(l-0.5))<0.001:
                # j = l-1/2
                defect = self.quantumDefect[0][l][0]+\
                    self.quantumDefect[0][l][1]/((n-self.quantumDefect[0][l][0])**2)+\
                    self.quantumDefect[0][l][2]/((n-self.quantumDefect[0][l][0])**4)+\
                    self.quantumDefect[0][l][3]/((n-self.quantumDefect[0][l][0])**6)+\
                    self.quantumDefect[0][l][4]/((n-self.quantumDefect[0][l][0])**8)+\
                    self.quantumDefect[0][l][5]/((n-self.quantumDefect[0][l][0])**10)
            else:
                # j = l + 1/2
                defect = self.quantumDefect[1][l][0]+\
                    self.quantumDefect[1][l][1]/((n-self.quantumDefect[1][l][0])**2)+\
                    self.quantumDefect[1][l][2]/((n-self.quantumDefect[1][l][0])**4)+\
                    self.quantumDefect[1][l][3]/((n-self.quantumDefect[1][l][0])**6)+\
                    self.quantumDefect[1][l][4]/((n-self.quantumDefect[1][l][0])**8)+\
                    self.quantumDefect[1][l][5]/((n-self.quantumDefect[1][l][0])**10)
        return defect

    
    def getReducedMatrixElementJ_asymmetric(self,n1,l1,j1,n2,l2,j2,s=0.5):
        """
            Reduced matrix element in :math:`J` basis, defined in asymmetric
            notation.

            Note that notation for symmetric and asymmetricly defined
            reduced matrix element is not consistent in the literature. For
            example, notation is used e.g. in Steck [1]_ is precisely the oposite.

            Note:
                Note that this notation is asymmetric: :math:`( j||e r \
                ||j' ) \\neq ( j'||e r ||j )`.
                Relation between the two notation is :math:`\\langle j||er||j'\\rangle=\
                \\sqrt{2j+1} ( j ||er ||j')`.
                This function always returns value for transition from
                lower to higher energy state, independent of the order of states
                entered in the function call.

            Args:
                n1 (int): principal quantum number of state 1
                l1 (int): orbital angular momentum of state 1
                j1 (float): total angular momentum of state 1
                n2 (int): principal quantum number of state 2
                l2 (int): orbital angular momentum of state 2
                j2 (float): total angular momentum of state 2

            Returns:
                float:
                    reduced dipole matrix element in Steck notation
                    :math:`( j || er || j' )` (:math:`a_0 e`).

            .. [1] Daniel A. Steck, "Cesium D Line Data," (revision 2.0.1, 2 May 2008).
                http://steck.us/alkalidata
        """
        #
        if (self.getTransitionFrequency(n1, l1, j1, n2, l2, j2)<0):
            temp = n2
            n2 = n1
            n1 = temp
            temp = l1
            l1 = l2
            l2 = temp
            temp = j1
            j1 = j2
            j2 = temp
        #Can I take this out?    
        #s = round(float((l1-l2+1.0))/2.0+j2+l1+1.0+0.5)
        return (-1)**(int((l2+l1+3.)/2.+s+j2))*\
                sqrt((2.0*j2+1.0)*(2.0*l1+1.0))*\
                Wigner6j(l1,l2,1,j2,j1,s)*\
                sqrt(float(max(l1,l2))/(2.0*l1+1.0))*\
                self.getRadialMatrixElement(n1, l1, j1, n2, l2, j2)

    def getReducedMatrixElementL(self,n1,l1,j1,n2,l2,j2,s=0.5):
        """
            Reduced matrix element in :math:`L` basis (symmetric notation)

            Args:
                n1 (int): principal quantum number of state 1
                l1 (int): orbital angular momentum of state 1
                j1 (float): total angular momentum of state 1
                n2 (int): principal quantum number of state 2
                l2 (int): orbital angular momentum of state 2
                j2 (float): total angular momentum of state 2

            Returns:
                float:
                    reduced dipole matrix element in :math:`L` basis
                    :math:`\\langle l || er || l' \\rangle` (:math:`a_0 e`).
        """
     
        r=  self.getRadialMatrixElement(n1, l1, j1, n2, l2, j2,s)

        return (-1)**l1*sqrt((2.0*l1+1.0)*(2.0*l2+1.0))*\
                Wigner3j(l1,1,l2,0,0,0)*r

    def getReducedMatrixElementJ(self,n1,l1,j1,n2,l2,j2,s=0.5):
        """
            Reduced matrix element in :math:`J` basis (symmetric notation)

            Args:
                n1 (int): principal quantum number of state 1
                l1 (int): orbital angular momentum of state 1
                j1 (float): total angular momentum of state 1
                n2 (int): principal quantum number of state 2
                l2 (int): orbital angular momentum of state 2
                j2 (float): total angular momentum of state 2

            Returns:
                float:
                    reduced dipole matrix element in :math:`J` basis
                    :math:`\\langle j || er || j' \\rangle` (:math:`a_0 e`).
        """

        return (-1)**(int(l1+s+j2+1.))*sqrt((2.*j1+1.)*(2.*j2+1.))*\
                Wigner6j(j1, 1., j2, l2, s, l1)*\
                self.getReducedMatrixElementL(n1,l1,j1,n2,l2,j2,s)


    def getDipoleMatrixElement(self,n1,l1,j1,mj1,n2,l2,j2,mj2,q,s = 0.5):
        """
            Dipole matrix element
            :math:`\\langle n_1 l_1 j_1 m_{j_1} |e\\mathbf{r}|n_2 l_2 j_2 m_{j_2}\\rangle`
            in units of :math:`a_0 e`

            Returns:
                float: dipole matrix element( :math:`a_0 e`)

            Example:

                For example, calculation of :math:`5 S_{1/2}m_j=-\\frac{1}{2} \\rightarrow  5 P_{3/2}m_j=-\\frac{3}{2}`
                transition dipole matrix element for laser driving :math:`\sigma^-`
                transition::

                    from arc import *
                    atom = Rubidium()
                    # transition 5 S_{1/2} m_j=-0.5 -> 5 P_{3/2} m_j=-1.5 for laser
                    # driving sigma- transition
                    print(atom.getDipoleMatrixElement(5,0,0.5,-0.5,5,1,1.5,-1.5,-1))


        """
        if abs(q)>1.1:
            return 0
        return (-1)**(int(j1-mj1))*\
                Wigner3j(j1, 1, j2, -mj1, -q, mj2)*\
                self.getReducedMatrixElementJ(n1,l1,j1,n2,l2,j2,s)


    def getRabiFrequency(self,n1,l1,j1,mj1,n2,l2,j2,q,laserPower,laserWaist,s=0.5):
        """
            Returns a Rabi frequency for resonantly driven atom in a
            center of TEM00 mode of a driving field

            Args:
                n1,l1,j1,mj1 : state from which we are driving transition
                n2,l2,j2 : state to which we are driving transition
                q : laser polarization (-1,0,1 correspond to :math:`\sigma^-`,
                    :math:`\pi` and :math:`\sigma^+` respectively)
                laserPower : laser power in units of W
                laserWaist : laser :math:`1/e^2` waist (radius) in units of m


            Returns:
                float:
                    Frequency in rad :math:`^{-1}`. If you want frequency in Hz,
                    divide by returned value by :math:`2\pi`
        """
        maxIntensity = 2*laserPower/(pi* laserWaist**2)
        electricField = sqrt(2.*maxIntensity/(C_c*epsilon_0))
        return self.getRabiFrequency2(n1,l1,j1,mj1,n2,l2,j2,q,electricField,s)

    def getRabiFrequency2(self,n1,l1,j1,mj1,n2,l2,j2,q,electricFieldAmplitude,s=0.5):
        """
            Returns a Rabi frequency for resonant excitation with a given
            electric field amplitude

            Args:
                n1,l1,j1,mj1 : state from which we are driving transition
                n2,l2,j2 : state to which we are driving transition
                q : laser polarization (-1,0,1 correspond to :math:`\sigma^-`,
                    :math:`\pi` and :math:`\sigma^+` respectively)
                electricFieldAmplitude : amplitude of electric field driving (V/m)

            Returns:
                float:
                    Frequency in rad :math:`^{-1}`. If you want frequency in Hz,
                    divide by returned value by :math:`2\pi`
        """
        mj2 = mj1+q
        if abs(mj2)-0.1>j2:
            return 0
        dipole = self.getDipoleMatrixElement(n1,l1,j1,mj1,n2,l2,j2,mj2,q,s)*\
                C_e*physical_constants["Bohr radius"][0]
        freq = electricFieldAmplitude*abs(dipole)/hbar
        return freq
    
    def getTransitionRate(self,n1,l1,j1,n2,l2,j2,temperature = 0.):
        """
            Transition rate due to coupling to vacuum modes (black body included)

            Calculates transition rate from the first given state to the second
            given state :math:`|n_1,l_1,j_1\\rangle \\rightarrow \
            |n_2,j_2,j_2\\rangle` at given temperature due to interaction with
            the vacuum field. For zero temperature this returns Einstein A
            coefficient. For details of calculation see Ref. [#lf1]_ and Ref. [#lf2]_.
            See `Black-body induced population transfer example snippet`_.

            .. _`Black-body induced population transfer example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-Atom-Lifetimes

            Args:
                n1 (int): principal quantum number
                l1 (int): orbital angular momentum
                j1 (float): total angular momentum
                n2 (int): principal quantum number
                l2 (int): orbital angular momentum
                j2 (float): total angular momentum
                [temperature] (float): temperature in K

            Returns:
                float:  transition rate in s :math:`{}^{-1}` (SI)

            References:
                .. [#lf1] C. E. Theodosiou, PRA **30**, 2881 (1984)
                    https://doi.org/10.1103/PhysRevA.30.2881

                .. [#lf2] I. I. Beterov, I. I. Ryabtsev, D. B. Tretyakov,\
                    and V. M. Entin, PRA **79**, 052504 (2009)
                    https://doi.org/10.1103/PhysRevA.79.052504
        """

        degeneracyTerm = 1.

        dipoleRadialPart = 0.0
        if (self.getTransitionFrequency(n1, l1, j1, n2, l2, j2)>0):
            dipoleRadialPart = self.getReducedMatrixElementJ_asymmetric(n1, l1, j1,\
                                                                        n2, l2, j2)*\
                                C_e*(physical_constants["Bohr radius"][0])

        else:
            #LIZZY: how come we are using the get redcued matrix element
            dipoleRadialPart = self.getReducedMatrixElementJ_asymmetric(n2, l2, j2,\
                                                                        n1, l1, j1)*\
                                C_e*(physical_constants["Bohr radius"][0])
            degeneracyTerm = (2.*j2+1.0)/(2.*j1+1.)

        omega = abs(2.0*pi*self.getTransitionFrequency(n1, l1, j1, n2, l2, j2))

        modeOccupationTerm = 0.
        if (self.getTransitionFrequency(n1, l1, j1, n2, l2, j2)<0):
            modeOccupationTerm = 1.

        # only possible by absorbing thermal photons ?
        if (hbar*omega < 100*C_k*temperature) and (omega > 1e2):
            modeOccupationTerm += 1./(exp(hbar*omega/(C_k*temperature))-1.)

        return omega**3*dipoleRadialPart**2/\
            (3*pi*epsilon_0*hbar*C_c**3)\
            *degeneracyTerm*modeOccupationTerm

    def getStateLifetime(self,n,l,j,temperature=0,includeLevelsUpTo = 0):
        """
            Returns the lifetime of the state (in s)

            For non-zero temperatures, user must specify up to which principal
            quantum number levels, that is **above** the initial state, should be
            included in order to account for black-body induced transitions to
            higher lying states. See `Rydberg lifetimes example snippet`_.

            .. _`Rydberg lifetimes example snippet`:
                ./Rydberg_atoms_a_primer.html#Rydberg-Atom-Lifetimes

            Args:
                n, l, j (int,int,float): specifies state whose lifetime we are calculating
                temperature : optional. Temperature at which the atom environment
                    is, measured in K. If this parameter is non-zero, user has
                    to specify transitions up to which state (due to black-body
                    decay) should be included in calculation.
                includeLevelsUpTo (int): optional and not needed for atom lifetimes
                    calculated at zero temperature. At non zero temperatures,
                    this specify maximum principal quantum number of the state
                    to which black-body induced transitions will be included.
                    Minimal value of the parameter in that case is :math:`n+1`


            Returns:
                float:
                    State lifetime in units of s (seconds)

            See also:
                :obj:`getTransitionRate` for calculating rates of individual
                transition rates between the two states

        """
        if (temperature>0.1 and includeLevelsUpTo<=n):
            raise ValueError("For non-zero temperatures, user has to specify \
            principal quantum number of the maximum state *above* the state for\
             which we are calculating the lifetime. This is in order to include \
             black-body induced transitions to higher lying up in energy levels.")
        elif (temperature<0.1):
            includeLevelsUpTo = max(n, self.groundStateN)

        transitionRate = 0.

        for nto in xrange(max(self.groundStateN,l),includeLevelsUpTo+1):

            # sum over all l-1
            if l>0:
                lto = l-1
                if lto > j-0.5-0.1:
                    jto = j
                    transitionRate += self.getTransitionRate(n,l,j,\
                                                            nto, lto, jto,\
                                                            temperature)
                jto = j-1.
                if jto>0:
                    transitionRate += self.getTransitionRate(n,l,j,\
                                                             nto, lto, jto,\
                                                             temperature)

        for nto in xrange(max(self.groundStateN,l+2),includeLevelsUpTo+1):
            # sum over all l+1
            lto = l+1
            if lto -0.5-0.1< j :
                jto = j
                transitionRate += self.getTransitionRate(n,l,j,\
                                                             nto, lto, jto,\
                                                             temperature)
            jto = j+1
            transitionRate += self.getTransitionRate(n,l,j,\
                                                    nto, lto, jto,\
                                                    temperature)
        # sum over additional states
        for state in self.extraLevels:
            if (abs(j-state[2])<0.6) and (state[2]!= l):
                transitionRate += self.getTransitionRate(n,l,j,\
                                                    state[0],state[1],state[2],\
                                                    temperature)

        return 1./transitionRate


    def getAverageSpeed(self,temperature):
        """
            Average (mean) speed at a given temperature

            Args:
                temperature (float): temperature (K)

            Returns:
                float: mean speed (m/s)
        """
        return sqrt( 8.*C_k*temperature/(pi*self.mass) )

    

    def getZeemanEnergyShift(self, l, j, mj, magneticFieldBz):
        """
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


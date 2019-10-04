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
                    s = int(row[6])
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
                    typeOfSource = int(row[10])  # 0 = experiment; 1 = theory
                    errorEstimate = float(row[11])
                    ref = row[12]
                    refdoi = row[13]
                    print(dme,comment,typeOfSource,errorEstimate,ref,refdoi)

                    literatureDME.append([n1,l1,j1*2,n2,l2,j2*2,s,dme,\
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
        # l2 = ? AND j2_x2 = ? AND j2_mj2 = ? AND s = ?
        # ''',(l1,2*j1,2*j1+mj1,l2,j2*2,2*j2+mj2,s))
        #answer = self.c.fetchone()
        #if (answer):
        #    return answer[0]

        #have just rep;aced all 0.5 with self.s, if we need to change it back we can 
        # calulates sum (See PRA 20:2251 (1979), eq.(10))
        sumPart = 0.
        if self.s == 0.5:
            ml = mj1 + self.s
            if (abs(ml)-0.1<l1)and(abs(ml)-0.1<l2):
    
                angularPart = 0.
                if (abs(l1-l2-1)<0.1):
                    angularPart = ((l1**2-ml**2)/((2.*l1+1.)*(2.*l1-1.)))**0.5
                elif(abs(l1-l2+1)<0.1):
                    angularPart = ((l2**2-ml**2)/((2.*l2+1.)*(2.*l2-1.)))**0.5
                #print(angularPart)
                sumPart += CG(l1,ml,self.s,mj1-ml,j1,mj1)*CG(l2,ml,self.s,mj1-ml,j2,mj2)*\
                            angularPart
    
            #LIZZY maybe we should not do this if we are s = 0
           
            ml = mj1 - self.s
            if (abs(ml)-0.1<l1)and(abs(ml)-0.1<l2):
                angularPart = 0.
                if (abs(l1-l2-1)<0.1):
                    angularPart = ((l1**2-ml**2)/((2.*l1+1.)*(2.*l1-1.)))**0.5
                elif(abs(l1-l2+1)<0.1):
                    angularPart = ((l2**2-ml**2)/((2.*l2+1.)*(2.*l2-1.)))**0.5
                sumPart += CG(l1,ml,self.s,mj1-ml,j1,mj1)*CG(l2,ml,self.s,mj1-ml,j2,mj2)*\
                            angularPart
                            
        else:
            for i in range(-self.s, self.s+1):
                ml = mj1 + i
                if (abs(ml)-0.1<l1)and(abs(ml)-0.1<l2):
                    angularPart = 0.
                    if (abs(l1-l2-1)<0.1):
                        angularPart = ((l1**2-ml**2)/((2.*l1+1.)*(2.*l1-1.)))**0.5
                    elif(abs(l1-l2+1)<0.1):
                        angularPart = ((l2**2-ml**2)/((2.*l2+1.)*(2.*l2-1.)))**0.5
                    sumPart += CG(l1,ml,self.s,mj1-ml,j1,mj1)*CG(l2,ml,self.s,mj1-ml,j2,mj2)*\
                                angularPart

        #self.c.execute(''' INSERT INTO eFieldCoupling_angular
        #                    VALUES (?,?,?, ?,?,?, ?,?)''',\
        #                    [l1,2*j1,2*j1+mj1,l2,j2*2,2*j2+mj2,sumPart,s] )
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

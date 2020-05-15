# -*- coding: utf-8 -*-

"""
    Level diagram calculations for atoms dressed by rydberg levels. 
    The dressing is achieved by a AC electromagnetic field (laser). 
    
    Most of the code here is from the module calculations_atom_pairstate.py. 
    This one add the AC field and the ground state to the Hamiltonian and diagonalizes it. 

    Example:
        Calculation of the eigenstates when the laser light is near resonant with the transition
        :math:`|~5~P_{3/2}~m_j=1/2\\rangle` -> `|60~S_{1/2}~m_j=1/2\\rangle` state. Colour
        highlights the mixture of state :math:`|~5~P_{3/2}~m_j=1/2\\rangle`:
            
            import arc as ARC
            n0=5;l0=1;j0=1.5;mj0=0.5; #Ground State
            nr=60;lr=0;jr=0.5;mjr=0.5; #Target rydberg State
            theta=0; #Polar Angle [0-pi]
            phi=0; #Azimuthal Angle [0-2pi]
            dn = 3; #Range of n to consider (n0-dn:n0+dn)
            dl = 3; #Range of l values
            deltaMax = 20e9 #Max pair-state energy difference [Hz]
            calc = ARC.DressedPairStateInteractions(ARC.Rubidium(), n0,l0,j0,nr,lr,jr, mj0,mjr,interactionsUpTo = 2, Omega0 = 8e-3,Delta0 = 30e-3)
            #Omega0 is the rabi frquency of the ac field and Delta0 is the detuning of the ac field from the transition.
            rvdw = calc.getLeRoyRadius()
            print("LeRoy radius = %.1f mum" % rvdw)
            #R array (um)
            r=np.linspace(1.5,10,1000)
            
            #Generate pair-state interaction Hamiltonian
            calc.defineBasis(theta,phi, dn,dl, deltaMax,progressOutput=True)
            #Diagonalise
            nEig=1 #Number of eigenstates to extract (we just want the ground state here)
            calc.diagonalise(r,nEig,progressOutput=True,sortEigenvectors = True)
            
            #Save data
            calc.exportData('60S_dressed_pair_calculation', exportFormat='csv') 
            
            #Plot 
            calc.plotLevelDiagram(hlim = [0.95,1])
            calc.ax.set_xlim(1.0,10.0)
            calc.ax.set_ylim(-5,3)
            calc.showPlot()  

"""

from __future__ import division, print_function, absolute_import

from .wigner import Wigner6j, Wigner3j, CG, WignerDmatrix
from .alkali_atom_functions import _EFieldCoupling, _atomLightAtomCoupling
from scipy.constants import physical_constants, pi, epsilon_0, hbar
import gzip
import sys
import datetime
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from .calculations_atom_single import StarkMap
from .alkali_atom_functions import *
from .divalent_atom_functions import DivalentAtom
from scipy.special import factorial
from scipy import floor
from scipy.special.specfun import fcoef
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, hstack, vstack
from numpy.lib.polynomial import real
from numpy.ma import conjugate
from scipy.optimize import curve_fit
from scipy.constants import e as C_e
from scipy.constants import h as C_h
from scipy.constants import c as C_c
from scipy.constants import k as C_k
import re
import numpy as np
from math import exp, log, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'


# for matrices


if sys.version_info > (2,):
    xrange = range


DPATH = os.path.join(os.path.expanduser('~'), '.arc-data')

#
class DressedPairStateInteractions:
    """
        Calculates level diagram (spaghetti) for levels of atoms dressed by rydberg state.
        
        Initializes Rydberg level spaghetti calculation for the given atom
        species (or for two atoms of different species) in the vicinity
        of the given pair state to which a laser light. For details of calculation see
        Ref. [?]_. 
        
        Args:
            atom (:obj:`AlkaliAtom` or :obj:`DivalentAtom`): = {
                :obj:`arc.alkali_atom_data.Lithium6`,
                :obj:`arc.alkali_atom_data.Lithium7`,
                :obj:`arc.alkali_atom_data.Sodium`,
                :obj:`arc.alkali_atom_data.Potassium39`,
                :obj:`arc.alkali_atom_data.Potassium40`,
                :obj:`arc.alkali_atom_data.Potassium41`,
                :obj:`arc.alkali_atom_data.Rubidium85`,
                :obj:`arc.alkali_atom_data.Rubidium87`,
                :obj:`arc.alkali_atom_data.Caesium`,
                :obj:`arc.divalent_atom_data.Strontium88`,
                :obj:`arc.divalent_atom_data.Calcium40`
                :obj:`arc.divalent_atom_data.Ytterbium174` }
                Select the alkali metal for energy level
                diagram calculation
            n (int): principal quantum number for the ground state
            l (int): orbital angular momentum for the ground state
            j (float): total angular momentum for the ground state
            nn (int): principal quantum number for the rydberg state
            ll (int): orbital angular momentum for the rydberg state
            jj (float): total angular momentum for the rydberg state
            m1 (float): projection of the total angular momentum on z-axis
                for the ground state
            m2 (float): projection of the total angular momentum on z-axis
                for the rydberg state
            interactionsUpTo (int): Optional. If set to 1, includes only
                dipole-dipole interactions. If set to 2 includes interactions
                up to quadrupole-quadrupole. Default value is 1.
            s (float): optional, spin state of the first atom. Default value
                of 0.5 is correct for :obj:`AlkaliAtom` but for
                :obj:`DivalentAtom` it has to be explicitly set to 0 or 1 for
                singlet and triplet states respectively.
                **If `s2` is not specified, it is assumed that the second
                atom is in the same spin state.**
            s2 (float): optinal, spin state of the second atom. If not
                specified (left to default value None) it will assume spin
                state of the first atom.
            atom2 (:obj:`AlkaliAtom` or :obj:`DivalentAtom`): optional,
                specifies atomic species for the second atom, enabeling
                calculation of **inter-species pair-state interactions**.
                If not specified (left to default value None) it will assume
                spin state of the first atom.

        References:
            .. [1] Jorge et al.

        Examples:
            **Advanced interfacing of pair-state is2=None, atom2=Nonenteractions calculations
            (PairStateInteractions class).** This
            is an advanced example intended for building up extensions to the
            existing code. If you want to directly access the pair-state
            interaction matrix, constructed by :obj:`defineBasis`,
            you can assemble it easily from diagonal part
            (stored in :obj:`matDiagonal` ) and off-diagonal matrices whose
            spatial dependence is :math:`R^{-3},R^{-4},R^{-5}` stored in that
            order in :obj:`matR`. Basis states are stored in :obj:`basisStates`
            array.

            >>> from arc import *
            >>> calc = PairStateInteractions(Rubidium(), 60,0,0.5, \
                60,0,0.5, 0.5,0.5,interactionsUpTo = 1)
            >>> # theta=0, phi = 0, range of pqn, range of l, deltaE = 25e9
            >>> calc.defineBasis(0 ,0 , 5, 5, 25e9, progressOutput=True)
            >>> # now calc stores interaction matrix and relevant basis
            >>> # we can access this directly and generate interaction matrix
            >>> # at distance rval :
            >>> rval = 4  # in mum
            >>> matrix = calc.matDiagonal
            >>> rX = (rval*1.e-6)**3
            >>> for matRX in self.matR:
            >>>     matrix = matrix + matRX/rX
            >>>     rX *= (rval*1.e-6)
            >>> # matrix variable now holds full interaction matrix for
            >>> # interacting atoms at distance rval calculated in
            >>> # pair-state basis states can be accessed as
            >>> basisStates = calc.basisStates
    """

    dataFolder = DPATH

    # =============================== Methods ===============================

    def __init__(self, atom, n, l, j, nn, ll, jj, m1, m2,
                 interactionsUpTo=1,
                 s=0.5,
                 s2=None, atom2=None, Omega0 = 0, Delta0 = 0):
        # alkali atom type, principal quantum number, orbital angular momentum,
        #  total angular momentum projections of the angular momentum on z axis
        self.atom1 = atom  #: atom type
        if atom2 is None:
            self.atom2 = atom
        else:
            self.atom2 = atom2
        self.n = n  # : pair-state definition: principal quantum number of the ground state
        self.l = l  # : pair-state definition: orbital angular momentum of the ground state
        self.j = j  # : pair-state definition: total angular momentum of the ground state
        self.nn = nn  # : pair-state definition: principal quantum number of rydberg state
        self.ll = ll  # : pair-state definition: orbital angular momentum of rydberg state
        self.jj = jj  # : pair-state definition: total angular momentum oof rydberg stateom
        self.m1 = m1  # : pair-state definition: projection of the total ang. momentum for the ground state
        self.m2 = m2  # : pair-state definition: projection of the total angular momentum for the rydberg state
        self.interactionsUpTo = interactionsUpTo
        """"
            Specifies up to which approximation we include in pair-state interactions.
            By default value is 1, corresponding to pair-state interactions up to
            dipole-dipole coupling. Value of 2 is also supported, corresponding
            to pair-state interactions up to quadrupole-quadrupole coupling.
        """
        self.Omega0 = Omega0 #Rabi frequency of the dressing with the near resonant transition (nn, ll, jj, m2).
        self.Delta0 = Delta0 # Deltuning from the near resonant transition (nn, ll, jj, m2)
        if (issubclass(type(atom),DivalentAtom) and not (s == 0 or s == 1)):
            raise ValueError("total angular spin s has to be defined explicitly "
                             "for calculations, and value has to be 0 or 1 "
                             "for singlet and tripplet states respectively.")
        self.s1 = s  #: total spin angular momentum, optional (default 0.5)

        if s2 is None:
            self.s2 = s
        else:
            self.s2 = s2

        # check that values of spin states are valid for entered atomic species

        if issubclass(type(self.atom1), DivalentAtom):
            if (abs(self.s1) > 0.1 and abs(self.s1 - 1) > 0.1):
                raise ValueError("atom1 is DivalentAtom and its spin has to be "
                                 "s=0 or s=1 (for singlet and triplet states "
                                 "respectively)")
        elif (abs(self.s1 - 0.5) > 0.1):
                raise ValueError("atom1 is AlkaliAtom and its spin has to be "
                                 "s=0.5")
        if issubclass(type(self.atom2), DivalentAtom):
            if (abs(self.s2) > 0.1 and abs(self.s2 - 1) > 0.1):
                raise ValueError("atom2 is DivalentAtom and its spin has to be "
                                 "s=0 or s=1 (for singlet and triplet states "
                                 "respectively)")
        elif  (abs(self.s2 - 0.5) > 0.1):
            # we have divalent atom
            raise ValueError("atom2 is AlkaliAtom and its spin has to be "
                             "s=0.5")
        if (abs((self.s1-self.m1) % 1) > 0.1):
            raise ValueError("atom1 with spin s = %.1d cannot have m1 = %.1d"
                             % (self.s1, self.m1))
        if (abs((self.s2-self.m2) % 1) > 0.1):
            raise ValueError("atom2 with spin s = %.1d cannot have m2 = %.1d"
                             % (self.s2, self.m2))

        # ====================== J basis (not resolving mj) ===================

        self.coupling = []
        """
            List of matrices defineing coupling strengths between the states in
             J basis (not resolving :math:`m_j` ). Basis is given by
             :obj:`channel`. Used as intermediary for full interaction matrix
             calculation by :obj:`defineBasis`.
        """
        self.channel = []
        """
            states relevant for calculation, defined in J basis (not resolving
            :math:`m_j`. Used as intermediary for full interaction matrix
            calculation by :obj:`defineBasis`.
        """

        # ======================= Full basis (resolving mj) ===================

        self.basisStates = []
        """
            List of pair-states for calculation. In the form
            [[n1,l1,j1,mj1,n2,l2,j2,mj2], ...].
            Each state is an array [n1,l1,j1,mj1,n2,l2,j2,mj2] corresponding to
            :math:`|n_1,l_1,j_1,m_{j1},n_2,l_2,j_2,m_{j2}\\rangle` state.
            Calculated by :obj:`defineBasis`.
        """
        self.matrixElement = []
        """
            `matrixElement[i]` gives index of state in :obj:`channel` basis
            (that doesn't resolve :obj:`m_j` states), for the given index `i`
            of the state in :obj:`basisStates`  ( :math:`m_j` resolving) basis.
        """

        # variuos parts of interaction matrix in pair-state basis
        self.matDiagonal = []
        """
            Part of interaction matrix in pair-state basis that doesn't depend
            on inter-atomic distance. E.g. diagonal elements of the interaction
            matrix, that describe energies of the pair states in unperturbed
            basis, will be stored here. Basis states are stored in
            :obj:`basisStates`. Calculated by :obj:`defineBasis`.
        """
        self.matR = []
        """
            Stores interaction matrices in pair-state basis
            that scale as :math:`1/R^3`, :math:`1/R^4` and :math:`1/R^5`
            with distance in  :obj:`matR[0]`, :obj:`matR[1]` and :obj:`matR[2]`
            respectively. These matrices correspond to dipole-dipole
            ( :math:`C_3`), dipole-quadrupole ( :math:`C_4`) and
            quadrupole-quadrupole ( :math:`C_5`) interactions
            coefficients. Basis states are stored in :obj:`basisStates`.
            Calculated by :obj:`defineBasis`.
        """
        self.originalPairStateIndex = 0
        """
            index of the original n,l,j,m1,nn,ll,jj,m2 pair-state in the
            :obj:`basisStates` basis.
        """

        self.matE = []
        self.matB_1 = []
        self.matB_2 = []

        # ===================== Eigen states and plotting =====================

        # finding perturbed energy levels
        self.r = []    # detuning scale
        self.y = []    # energy levels
        self.highlight = []

        # pointers towards figure
        self.fig = 0
        self.ax = 0

        # for normalization of the maximum coupling later
        self.maxCoupling = 0.

        # n,l,j,mj, drive polarization q
        self.drivingFromState = [0, 0, 0, 0, 0]

        # sam = saved angular matrix metadata
        self.angularMatrixFile = "angularMatrix.npy"
        self.angularMatrixFile_meta = "angularMatrix_meta.npy"
        #self.sam = []
        self.savedAngularMatrix_matrix = []

        # intialize precalculated values for factorial term
        # in __getAngularMatrix_M
        def fcoef(l1, l2, m):
            return factorial(l1 + l2) / (factorial(l1 + m)
                                         * factorial(l1 - m)
                                         * factorial(l2 + m)
                                         * factorial(l2 - m))**0.5
        x = self.interactionsUpTo
        self.fcp = np.zeros((x + 1, x + 1, 2 * x + 1))
        for c1 in range(1, x + 1):
            for c2 in range(1, x + 1):
                for p in range(-min(c1, c2), min(c1, c2) + 1):
                    self.fcp[c1, c2, p + x] = fcoef(c1, c2, p)

        self.conn = False
        self.c = False

    def __getAngularMatrix_M(self, l, j, ll, jj, l1, j1, l2, j2):
        # did we already calculated this matrix?

        self.c.execute('''SELECT ind FROM pair_angularMatrix WHERE
             l1 = ? AND j1_x2 = ? AND
             l2 = ? AND j2_x2 = ? AND
             l3 = ? AND j3_x2 = ? AND
             l4 = ? AND j4_x2 = ?
             ''', (l, j * 2, ll, jj * 2, l1, j1 * 2, l2, j2 * 2))

        index = self.c.fetchone()
        if (index):
            return self.savedAngularMatrix_matrix[index[0]]

        # determine coupling
        dl = abs(l - l1)
        dj = abs(j - j1)
        c1 = 0
        if dl == 1 and (dj < 1.1):
            c1 = 1  # dipole coupling
        elif (dl == 0 or dl == 2 or dl == 1):
            c1 = 2  # quadrupole coupling
        else:
            raise ValueError("error in __getAngularMatrix_M")
            exit()
        dl = abs(ll - l2)
        dj = abs(jj - j2)
        c2 = 0
        if dl == 1 and (dj < 1.1):
            c2 = 1  # dipole coupling
        elif (dl == 0 or dl == 2 or dl == 1):
            c2 = 2  # quadrupole coupling
        else:
            raise ValueError("error in __getAngularMatrix_M")
            exit()

        am = np.zeros((int(round((2 * j1 + 1) * (2 * j2 + 1), 0)),
                       int(round((2 * j + 1) * (2 * jj + 1), 0))),
                      dtype=np.float64)

        if (c1 > self.interactionsUpTo) or (c2 > self.interactionsUpTo):
            return am

        j1range = np.linspace(-j1, j1, round(2 * j1) + 1)
        j2range = np.linspace(-j2, j2, round(2 * j2) + 1)
        jrange = np.linspace(-j, j, int(2 * j) + 1)
        jjrange = np.linspace(-jj, jj, int(2 * jj) + 1)

        for m1 in j1range:
            for m2 in j2range:
                # we have chosen the first index
                index1 = int(round(m1 * (2.0 * j2 + 1.0) + m2
                                   + (j1 * (2.0 * j2 + 1.0) + j2), 0))
                for m in jrange:
                    for mm in jjrange:
                        # we have chosen the second index
                        index2 = int(round(m * (2.0 * jj + 1.0)
                                           + mm + (j * (2.0 * jj + 1.0) + jj),
                                           0)
                                     )

                        # angular matrix element from Sa??mannshausen, Heiner,
                        # Merkt, Fr??d??ric, Deiglmayr, Johannes
                        # PRA 92: 032505 (2015)
                        elem = (-1.0)**(j + jj + self.s1 + self.s2 + l1 + l2) * \
                            CG(l, 0, c1, 0, l1, 0) * CG(ll, 0, c2, 0, l2, 0)
                        elem = elem * \
                            sqrt((2.0 * l + 1.0) * (2.0 * ll + 1.0)) * \
                            sqrt((2.0 * j + 1.0) * (2.0 * jj + 1.0))
                        elem = elem * \
                            Wigner6j(l, self.s1, j, j1, c1, l1) * \
                            Wigner6j(ll, self.s2, jj, j2, c2, l2)

                        sumPol = 0.0  # sum over polarisations
                        limit = min(c1, c2)
                        for p in xrange(-limit, limit + 1):
                            sumPol = sumPol + \
                                self.fcp[c1, c2, p + self.interactionsUpTo] * \
                                CG(j, m, c1, p, j1, m1) *\
                                CG(jj, mm, c2, -p, j2, m2)
                        am[index1, index2] = elem * sumPol

        index = len(self.savedAngularMatrix_matrix)

        self.c.execute(''' INSERT INTO pair_angularMatrix
                            VALUES (?,?, ?,?, ?,?, ?,?, ?)''',
                       (l, j * 2, ll, jj * 2, l1, j1 * 2, l2, j2 * 2, index))
        self.conn.commit()

        self.savedAngularMatrix_matrix.append(am)
        self.savedAngularMatrixChanged = True

        return am

    def __updateAngularMatrixElementsFile(self):
        if not (self.savedAngularMatrixChanged):
            return

        try:
            self.c.execute('''SELECT * FROM pair_angularMatrix ''')
            data = []
            for v in self.c.fetchall():
                data.append(v)

            data = np.array(data, dtype=np.float32)

            data[:, 1] /= 2.  # 2 r j1 -> j1
            data[:, 3] /= 2.  # 2 r j2 -> j2
            data[:, 5] /= 2.  # 2 r j3 -> j3
            data[:, 7] /= 2.  # 2 r j4 -> j4

            fileHandle = gzip.GzipFile(
                os.path.join(self.dataFolder, self.angularMatrixFile_meta),
                'wb'
                )
            np.save(fileHandle, data)
            fileHandle.close()
        except IOError as e:
            print("Error while updating angularMatrix \
                data meta (description) File " + self.angularMatrixFile_meta)

        try:
            fileHandle = gzip.GzipFile(
                os.path.join(self.dataFolder, self.angularMatrixFile),
                'wb'
                )
            np.save(fileHandle, self.savedAngularMatrix_matrix)
            fileHandle.close()
        except IOError as e:
            print("Error while updating angularMatrix \
                    data File " + self.angularMatrixFile)
            print(e)

    def __loadAngularMatrixElementsFile(self):
        try:
            fileHandle = gzip.GzipFile(
                os.path.join(self.dataFolder, self.angularMatrixFile_meta),
                'rb'
                )
            data = np.load(fileHandle, encoding='latin1', allow_pickle=True)
            fileHandle.close()
        except:
            print("Note: No saved angular matrix metadata files to be loaded.")
            print(sys.exc_info())
            return

        data[:, 1] *= 2  # j1 -> 2 r j1
        data[:, 3] *= 2  # j2 -> 2 r j2
        data[:, 5] *= 2  # j3 -> 2 r j3
        data[:, 7] *= 2  # j4 -> 2 r j4

        data = np.array(np.rint(data), dtype=np.int)

        try:

            self.c.executemany('''INSERT INTO pair_angularMatrix
                (l1, j1_x2 ,
                 l2 , j2_x2 ,
                 l3, j3_x2,
                 l4 , j4_x2 ,
                 ind)
                      VALUES (?,?,?,?,?,?,?,?,?)''', data)

            self.conn.commit()

        except sqlite3.Error as e:
            print("Error while loading precalculated values into the database!")
            print(e)
            exit()
        if len(data) == 0:
            print("error")
            return

        try:
            fileHandle = gzip.GzipFile(
                os.path.join(self.dataFolder, self.angularMatrixFile),
                'rb'
                )
            self.savedAngularMatrix_matrix = np.load(
                fileHandle,
                encoding='latin1',
                allow_pickle=True).tolist()
            fileHandle.close()
        except:
            print("Note: No saved angular matrix files to be loaded.")
            print(sys.exc_info())

    def __isCoupled(self, n, l, j, nn, ll, jj, n1, l1, j1, n2, l2, j2, limit):
        if ((abs(self.__getEnergyDefect(n, l, j,
                                        nn, ll, jj,
                                        n1, l1, j1,
                                        n2, l2, j2)
                 ) / C_h < limit)
            and not (n == n1 and nn == n2
                     and l == l1 and ll == l2
                     and j == j1 and jj == j2)
                and not ((abs(l1 - l) != 1
                          and( (abs(j - 0.5) < 0.1
                                and abs(j1 - 0.5) < 0.1) # j = 1/2 and j'=1/2 forbidden
                                or
                                (abs(j) < 0.1
                                and abs(j1 - 1) < 0.1)  # j = 0 and j'=1 forbidden
                                or
                                (abs(j-1) < 0.1
                                and abs(j1) < 0.1)  # j = 1 and j'=0 forbidden
                                )
                          )
                         or (abs(l2 - ll) != 1
                            and( (abs(jj - 0.5) < 0.1
                                and abs(j2 - 0.5) < 0.1) # j = 1/2 and j'=1/2 forbidden
                                or
                                (abs(jj) < 0.1
                                and abs(j2 - 1) < 0.1)  # j = 0 and j'=1 forbidden
                                or
                                (abs(jj-1) < 0.1
                                and abs(j2) < 0.1)  # j = 1 and j'=0 forbidden
                                )
                          )
                        )
                and not(abs(j)<0.1 and abs(j1)<0.1)  # j = 0 and j'=0 forbiden
                and not (abs(jj)<0.1 and abs(j2)<0.1)
                and not (abs(l)<0.1 and abs(l1)<0.1) # l = 0 and l' = 0 is forbiden
                and not (abs(ll)<0.1 and abs(l2)<0.1)
                ):
            # determine coupling
            dl = abs(l - l1)
            dj = abs(j - j1)
            c1 = 0
            if dl == 1 and (dj < 1.1):
                c1 = 1  # dipole coupling
            elif (dl == 0 or dl == 2 or dl == 1)and (dj < 2.1) and \
                    (2 <= self.interactionsUpTo):
                c1 = 2  # quadrupole coupling
            else:
                return False
            dl = abs(ll - l2)
            dj = abs(jj - j2)
            c2 = 0
            if dl == 1 and (dj < 1.1):
                c2 = 1  # dipole coupling
            elif (dl == 0 or dl == 2 or dl == 1) and (dj < 2.1) and \
                    (2 <= self.interactionsUpTo):
                c2 = 2  # quadrupole coupling
            else:
                return False
            return c1 + c2
        else:
            return False

    def __getEnergyDefect(self,
                          n, l, j,
                          nn, ll, jj,
                          n1, l1, j1,
                          n2, l2, j2):
        """
        Energy defect between |n,l,j>x|nn,ll,jj> state and |n1,l1,j1>x|n1,l1,j1>
        state of atom1 and atom2 in respective spins states s1 and s2

        Takes spin vales s1 and s2 as the one defined when defining calculation.

        Args:
            n (int): principal quantum number
            l (int): orbital angular momentum
            j (float): total angular momentum
            nn (int): principal quantum number
            ll (int): orbital angular momentum
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
        return C_e * (self.atom1.getEnergy(n1, l1, j1, s=self.s1)
                      + self.atom2.getEnergy(n2, l2, j2, s=self.s2)
                      - self.atom1.getEnergy(n, l, j, s=self.s1)
                      - self.atom2.getEnergy(nn, ll, jj, s=self.s2))

    def __makeRawMatrix2(self,
                         nn, ll, jj,
                         k, lrange, limit, limitBasisToMj,
                         progressOutput=False, debugOutput=False):
        
        
        n = nn 
        l = ll
        j = jj
         
        # limit = limit in Hz on energy defect
        # k defines range of n' = [n-k, n+k]
        dimension = 0

        # which states/channels contribute significantly in the second order perturbation?
        states = []

        # original pairstate index
        opi = 0

        # this numbers are conserved if we use only dipole-dipole interactions
        Lmod2 = ((l + ll) % 2)

        l1start = l - 1
        if l == 0:
            l1start = 0

        l2start = ll - 1
        if ll == 0:
            l2start = 0

        if debugOutput:
            print("\n ======= Relevant states =======\n")

        for n1 in xrange(max(n - k, 1), n + k + 1):
            for n2 in xrange(max(nn - k, 1), nn + k + 1):
                l1max = max(l + self.interactionsUpTo, lrange) + 1
                l1max = min(l1max, n1 - 1)
                for l1 in xrange(l1start, l1max):
                    l2max = max(ll + self.interactionsUpTo, lrange) + 1
                    l2max = min(l2max, n2 - 1)
                    for l2 in xrange(l2start, l2max):
                        j1 = l1 - self.s1
                        while j1 < -0.1:
                            j1 += 2 * self.s1
                        while j1 <= l1 + self.s1 + 0.1:
                            j2 = l2 - self.s2
                            while j2 < -0.1:
                                j2 += 2 * self.s2

                            while j2 <= l2 + self.s2 + 0.1:
                                ed = self.__getEnergyDefect(n, l, j,
                                                            nn, ll, jj,
                                                            n1, l1, j1,
                                                            n2, l2, j2) / C_h
                                if (abs(ed) < limit
                                    and (not (self.interactionsUpTo == 1)
                                         or (Lmod2 == ((l1 + l2) % 2)))
                                    and ((not limitBasisToMj)
                                         or (j1 + j2 + 0.1
                                             > self.m1 + self.m2))
                                    and (n1 >= self.atom1.groundStateN
                                         or [n1, l1, j1] in self.atom1.extraLevels)
                                    and (n2 >= self.atom2.groundStateN
                                        or [n2, l2, j2] in self.atom2.extraLevels)
                                        ):

                                    if debugOutput:
                                        pairState = (
                                            "|"
                                            + printStateString(n1, l1, j1,
                                                               s=self.s1)
                                            + ","
                                            + printStateString(n2, l2, j2,
                                                               s=self.s2)
                                            + ">")
                                        print(
                                            pairState
                                            + ("\t EnergyDefect = %.3f GHz"
                                               % (ed * 1.e-9)
                                              )
                                            )

                                    states.append([n1, l1, j1, n2, l2, j2])

                                    if (n == n1 and nn == n2
                                        and l == l1 and ll == l2
                                        and j == j1 and jj == j2
                                            ):
                                        opi = dimension

                                    dimension = dimension + 1
                                j2 = j2 + 1.0
                            j1 = j1 + 1.0

        if debugOutput:
            print("\tMatrix dimension\t=\t", dimension)
        m = np.zeros((dimension, dimension), dtype=np.float64)

        # mat_value, mat_row, mat_column for each sparce matrix describing
        # dipole-dipole, dipole-quadrupole (and quad-dipole) and quadrupole-quadrupole
        couplingMatConstructor = [[[], [], []]
                                  for i in xrange(2 * self.interactionsUpTo - 1)]

        # original pair-state (i.e. target pair state) Zeeman Shift
        opZeemanShift = (self.atom1.getZeemanEnergyShift(
            self.l, self.j, self.m1,
            self.Bz,
            s=self.s1)
            + self.atom2.getZeemanEnergyShift(
                self.ll, self.jj, self.m2,
                self.Bz,
                s=self.s2)
            ) / C_h * 1.0e-9  # in GHz

        if debugOutput:
            print("\n ======= Coupling strengths (radial part only) =======\n")

        maxCoupling = "quadrupole-quadrupole"
        if (self.interactionsUpTo == 1):
            maxCoupling = "dipole-dipole"
        if debugOutput:
            print("Calculating coupling (up to ",
                  maxCoupling, ") between the pair states")

        for i in xrange(dimension):

            ed = self.__getEnergyDefect(
                states[opi][0], states[opi][1], states[opi][2],
                states[opi][3], states[opi][4], states[opi][5],
                states[i][0], states[i][1], states[i][2],
                states[i][3], states[i][4], states[i][5]) / C_h * 1.0e-9\
                - opZeemanShift

            pairState1 = (
                "|"
                + printStateString(states[i][0], states[i][1], states[i][2],
                                   s=self.s1)
                + ","
                + printStateString(states[i][3], states[i][4], states[i][5],
                                   s=self.s2)
                + ">"
                )

            states[i].append(ed)  # energy defect of given state

            for j in xrange(i + 1, dimension):

                coupled = self.__isCoupled(
                    states[i][0], states[i][1], states[i][2],
                    states[i][3], states[i][4], states[i][5],
                    states[j][0], states[j][1], states[j][2],
                    states[j][3], states[j][4], states[j][5], limit)

                if (states[i][0] == 24 and states[j][0] == 18):
                    print("\n")
                    print(states[i])
                    print(states[j])
                    print(coupled)

                if coupled and (abs(states[i][0] - states[j][0]) <= k
                                and abs(states[i][3] - states[j][3]) <= k):
                    if debugOutput:
                        pairState2 = ("|"
                            + printStateString(states[j][0],
                                               states[j][1],
                                               states[j][2],
                                               s=self.s1)
                            + ","
                            + printStateString(states[j][3],
                                               states[j][4],
                                               states[j][5],
                                               s=self.s2)
                            + ">")
                        print(pairState1 + " <---> " + pairState2)

                    couplingStregth = _atomLightAtomCoupling(
                        states[i][0], states[i][1], states[i][2],
                        states[i][3], states[i][4], states[i][5],
                        states[j][0], states[j][1], states[j][2],
                        states[j][3], states[j][4], states[j][5],
                        self.atom1, atom2=self.atom2,
                        s=self.s1, s2=self.s2) / C_h * 1.0e-9

                    couplingMatConstructor[coupled - 2][0].append(
                        couplingStregth)
                    couplingMatConstructor[coupled - 2][1].append(i)
                    couplingMatConstructor[coupled - 2][2].append(j)

                    exponent = coupled + 1
                    if debugOutput:
                        print(("\tcoupling (C_%d/R^%d) = %.5f"
                               % (exponent, exponent,
                                  couplingStregth * (1e6)**(exponent))),
                              "/R^", exponent, " GHz  (mu m)^", exponent, "\n"
                              )

        # coupling = [1,1] dipole-dipole, [2,1]  quadrupole dipole, [2,2] quadrupole quadrupole

        couplingMatArray = [
            csr_matrix(
                (couplingMatConstructor[i][0],
                 (couplingMatConstructor[i][1], couplingMatConstructor[i][2])
                 ),
                shape=(dimension, dimension)
                )
            for i in xrange(len(couplingMatConstructor))
            ]
        return states, couplingMatArray

    def __initializeDatabaseForMemoization(self):
        # memoization of angular parts
        self.conn = sqlite3.connect(os.path.join(self.dataFolder,
                                                 "precalculated_pair.db"))
        self.c = self.conn.cursor()

        # ANGULAR PARTS
        self.c.execute('''DROP TABLE IF EXISTS pair_angularMatrix''')
        self.c.execute('''SELECT COUNT(*) FROM sqlite_master
                            WHERE type='table' AND name='pair_angularMatrix';''')
        if (self.c.fetchone()[0] == 0):
            # create table
            try:
                self.c.execute('''CREATE TABLE IF NOT EXISTS pair_angularMatrix
                 (l1 TINYINT UNSIGNED, j1_x2 TINYINT UNSIGNED,
                 l2 TINYINT UNSIGNED, j2_x2 TINYINT UNSIGNED,
                 l3 TINYINT UNSIGNED, j3_x2 TINYINT UNSIGNED,
                 l4 TINYINT UNSIGNED, j4_x2 TINYINT UNSIGNED,
                 ind INTEGER,
                 PRIMARY KEY (l1,j1_x2, l2,j2_x2, l3,j3_x2, l4,j4_x2)
                ) ''')
            except sqlite3.Error as e:
                print(e)
            self.conn.commit()
        self.__loadAngularMatrixElementsFile()
        self.savedAngularMatrixChanged = False

    def __closeDatabaseForMemoization(self):
        self.conn.commit()
        self.conn.close()
        self.conn = False
        self.c = False

    def getLeRoyRadius(self):
        """
            Returns Le Roy radius for initial pair-state.

            Le Roy radius [#leroy]_ is defined as
            :math:`2(\\langle r_1^2 \\rangle^{1/2} + \\langle r_2^2 \\rangle^{1/2})`,
            where :math:`r_1` and :math:`r_2` are electron coordinates for the
            first and the second atom in the initial pair-state.
            Below this radius, calculations are not valid since electron
            wavefunctions start to overlap.

            Returns:
                float: LeRoy radius measured in :math:`\\mu m`

            References:
                .. [#leroy] R.J. Le Roy, Can. J. Phys. **52**, 246 (1974)
                    http://www.nrcresearchpress.com/doi/abs/10.1139/p74-035
        """
        step = 0.001
        r1, psi1_r1 = self.atom2.radialWavefunction(
            self.ll, 0.5, self.jj,
            self.atom2.getEnergy(self.nn, self.ll, self.jj, s=self.s2) / 27.211,
            self.atom2.alphaC**(1 / 3.0),
            2.0 * self.nn * (self.nn + 15.0), step)

        sqrt_r1_on2 = np.trapz(np.multiply(np.multiply(psi1_r1, psi1_r1),
                                           np.multiply(r1, r1)),
                               x=r1)

        r2, psi2_r2 = self.atom2.radialWavefunction(
            self.ll, 0.5, self.jj,
            self.atom2.getEnergy(self.nn, self.ll, self.jj, s=self.s2) / 27.211,
            self.atom2.alphaC**(1 / 3.0),
            2.0 * self.nn * (self.nn + 15.0), step)

        sqrt_r2_on2 = np.trapz(np.multiply(np.multiply(psi2_r2, psi2_r2),
                                           np.multiply(r2, r2)),
                               x=r2)

        return 2. * (sqrt(sqrt_r1_on2) + sqrt(sqrt_r2_on2))\
            * (physical_constants["Bohr radius"][0] * 1.e6)



    def defineBasis(self, theta, phi, nRange, lrange, energyDelta,
                    Bz=0, progressOutput=False, debugOutput=False):
        r"""
            Finds relevant states in the vicinity of the given rydberg-level

            Finds relevant ryberg-level basis and calculates interaction matrix.
            ryberg-level basis is saved in :obj:`basisStates`.
            Interaction matrix is saved in parts depending on the scaling with
            distance. Diagonal elements :obj:`matDiagonal`, correponding to
            relative energy defects of the pair-states, don't change with
            interatomic separation. Off diagonal elements can depend
            on distance as :math:`R^{-3}, R^{-4}` or :math:`R^{-5}`,
            corresponding to dipole-dipole (:math:`C_3` ), dipole-qudrupole
            (:math:`C_4` ) and quadrupole-quadrupole coupling (:math:`C_5` )
            respectively. These parts of the matrix are stored in :obj:`matR`
            in that order. I.e. :obj:`matR[0]` stores dipole-dipole coupling
            (:math:`\propto R^{-3}`), :obj:`matR[0]` stores dipole-quadrupole
            couplings etc.

            Args:
                theta (float):  relative orientation of the two atoms
                    (see figure on top of the page), range 0 to :math:`\pi`
                phi (float): relative orientation of the two atoms (see figure
                    on top of the page), range 0 to :math:`2\pi`
                nRange (int): how much below and above the given principal
                    quantum number of the pair state we should be looking?
                lrange (int): what is the maximum angular orbital momentum
                    state that we are including in calculation
                energyDelta (float): what is maximum energy difference (
                    :math:`\Delta E/h` in Hz)
                    between the original pair state and the other pair states
                    that we are including in calculation
                Bz (float): optional, magnetic field directed along z-axis in
                    units of Tesla. Calculation will be correct only for weak
                    magnetic fields, where paramagnetic term is much stronger
                    then diamagnetic term. Diamagnetic term is neglected.
                progressOutput (bool): optional, False by default. If true,
                    prints information about the progress of the calculation.
                debugOutput (bool): optional, False by default. If true,
                    similarly to progressOutput=True, this will print
                    information about the progress of calculations, but with
                    more verbose output.

            See also:
                :obj:`alkali_atom_functions.saveCalculation` and
                :obj:`alkali_atom_functions.loadSavedCalculation` for
                information on saving intermediate results of calculation for
                later use.
        """

        self.__initializeDatabaseForMemoization()

        # save call parameters
        self.theta = theta
        self.phi = phi
        self.nRange = nRange
        self.lrange = lrange
        self.energyDelta = energyDelta
        self.Bz = Bz

        self.basisStates = []

        # wignerDmatrix
        wgd = WignerDmatrix(theta, phi)

        limitBasisToMj = False
        if (theta < 0.001):
            limitBasisToMj = True  # Mj will be conserved in calculations

        originalMj = self.m1 + self.m2

        self.channel, self.coupling = self.__makeRawMatrix2(
            self.nn, self.ll, self.jj,
            nRange, lrange, energyDelta,
            limitBasisToMj,
            progressOutput=progressOutput,
            debugOutput=debugOutput)

        self.atom1.updateDipoleMatrixElementsFile()
        self.atom2.updateDipoleMatrixElementsFile()

        # generate all the states (with mj principal quantum number)

        # opi = original pairstate index
        opi = 0

        # NEW FOR SPACE MATRIX
        self.index = np.zeros(len(self.channel) + 1, dtype=np.int16)

        for i in xrange(len(self.channel)):
            self.index[i] = len(self.basisStates)

            stateCoupled = self.channel[i]

            for m1c in np.linspace(stateCoupled[2], -stateCoupled[2],
                                   round(1 + 2 * stateCoupled[2])):
                for m2c in np.linspace(stateCoupled[5], -stateCoupled[5],
                                       round(1 + 2 * stateCoupled[5])):
                    if ((not limitBasisToMj) or (abs(originalMj
                                                     - m1c - m2c) < 0.1)):
                        self.basisStates.append(
                            [stateCoupled[0], stateCoupled[1], stateCoupled[2],
                             m1c,
                             stateCoupled[3], stateCoupled[4], stateCoupled[5],
                             m2c])
                        self.matrixElement.append(i)

                        if (abs(stateCoupled[0] - self.n) < 0.1
                            and abs(stateCoupled[1] - self.l) < 0.1
                            and abs(stateCoupled[2] - self.j) < 0.1
                            and abs(m1c - self.m1) < 0.1
                            and abs(stateCoupled[3] - self.nn) < 0.1
                            and abs(stateCoupled[4] - self.ll) < 0.1
                            and abs(stateCoupled[5] - self.jj) < 0.1
                            and abs(m2c - self.m2) < 0.1):
                            opi = len(self.basisStates) - 1
            if (self.index[i] == len(self.basisStates)):
                print(stateCoupled)
        self.index[-1] = len(self.basisStates)

        if progressOutput or debugOutput:
            print("\nCalculating Hamiltonian matrix...\n")

        dimension = len(self.basisStates)
        if progressOutput or debugOutput:
            print("\n\tmatrix (dimension ", dimension, ")\n")

        # INITIALIZING MATICES
        # all (sparce) matrices will be saved in csr format
        # value, row, column
        matDiagonalConstructor = [[], [], []]

        matRConstructor = [[[], [], []]
                           for i in xrange(self.interactionsUpTo * 2 - 1)]

        matRIndex = 0
        for c in self.coupling:
            progress = 0.
            for ii in xrange(len(self.channel)):
                if progressOutput:
                    dim = len(self.channel)
                    progress += ((dim - ii) * 2 - 1)
                    sys.stdout.write(
                        "\rMatrix R%d %.1f %% (state %d of %d)"
                        % (matRIndex + 3,
                           float(progress) / float(dim**2) * 100.,
                           ii + 1,
                           len(self.channel)))
                    sys.stdout.flush()

                ed = self.channel[ii][6]

                # solves problems with exactly degenerate basisStates
                degeneracyOffset = 0.00000001

                i = self.index[ii]
                dMatrix1 = wgd.get(self.basisStates[i][2])
                dMatrix2 = wgd.get(self.basisStates[i][6])

                for i in xrange(self.index[ii], self.index[ii + 1]):
                    statePart1 = singleAtomState(
                        self.basisStates[i][2], self.basisStates[i][3])
                    statePart2 = singleAtomState(
                        self.basisStates[i][6], self.basisStates[i][7])
                    # rotate individual states

                    statePart1 = dMatrix1.dot(statePart1)
                    statePart2 = dMatrix2.dot(statePart2)

                    stateCom = compositeState(statePart1, statePart2)

                    if (matRIndex == 0):
                        zeemanShift = (
                            self.atom1.getZeemanEnergyShift(
                                self.basisStates[i][1],
                                self.basisStates[i][2],
                                self.basisStates[i][3],
                                self.Bz,
                                s=self.s1)
                            + self.atom2.getZeemanEnergyShift(
                                self.basisStates[i][5],
                                self.basisStates[i][6],
                                self.basisStates[i][7],
                                self.Bz,
                                s=self.s2)
                            ) / C_h * 1.0e-9   # in GHz
                        matDiagonalConstructor[0].append(ed + zeemanShift
                                                         + degeneracyOffset)
                        degeneracyOffset += 0.00000001
                        matDiagonalConstructor[1].append(i)
                        matDiagonalConstructor[2].append(i)

                    for dataIndex in xrange(c.indptr[ii], c.indptr[ii + 1]):

                        jj = c.indices[dataIndex]
                        radialPart = c.data[dataIndex]

                        j = self.index[jj]
                        dMatrix3 = wgd.get(self.basisStates[j][2])
                        dMatrix4 = wgd.get(self.basisStates[j][6])

                        if (self.index[jj] != self.index[jj + 1]):
                            d = self.__getAngularMatrix_M(
                                self.basisStates[i][1], self.basisStates[i][2],
                                self.basisStates[i][5], self.basisStates[i][6],
                                self.basisStates[j][1], self.basisStates[j][2],
                                self.basisStates[j][5], self.basisStates[j][6])
                            secondPart = d.dot(stateCom)
                        else:
                            print(" - - - ", self.channel[jj])

                        for j in xrange(self.index[jj], self.index[jj + 1]):
                            statePart1 = singleAtomState(
                                self.basisStates[j][2], self.basisStates[j][3])
                            statePart2 = singleAtomState(
                                self.basisStates[j][6], self.basisStates[j][7])
                            # rotate individual states

                            statePart1 = dMatrix3.dot(statePart1)
                            statePart2 = dMatrix4.dot(statePart2)
                            # composite state of two atoms
                            stateCom2 = compositeState(statePart1, statePart2)

                            angularFactor = conjugate(
                                stateCom2.T).dot(secondPart)
                            angularFactor = real(angularFactor[0, 0])

                            if (abs(angularFactor) > 1.e-5):
                                matRConstructor[matRIndex][0].append(
                                    radialPart * angularFactor)
                                matRConstructor[matRIndex][1].append(i)
                                matRConstructor[matRIndex][2].append(j)

                                matRConstructor[matRIndex][0].append(
                                    radialPart * angularFactor)
                                matRConstructor[matRIndex][1].append(j)
                                matRConstructor[matRIndex][2].append(i)
            matRIndex += 1
            if progressOutput or debugOutput:
                print("\n")

        self.matDiagonal = csr_matrix(
            (matDiagonalConstructor[0],
             (matDiagonalConstructor[1], matDiagonalConstructor[2])),
            shape=(dimension, dimension)
            )

        self.matR = [
            csr_matrix((matRConstructor[i][0],
                        (matRConstructor[i][1], matRConstructor[i][2])),
                       shape=(dimension, dimension)
                       ) for i in xrange(self.interactionsUpTo * 2 - 1)
            ]

        self.originalPairStateIndex = opi

        self.__updateAngularMatrixElementsFile()
        self.__closeDatabaseForMemoization()
    
    def __getDressedMatrixElements(self,UNmat):
        r"""
            This part is an addition to the various functions defined in calculations_atom_pairstate.py
            We add the ground state and the states in which one atom from the pair is excited to the target 
            Rydberg state.See ref [?] for more details.

            Args:
                Hamiltonian matrix with all rydberg pairs and thier interactions. 
            Returns:
                Hamiltonian matrix with the ground state and the intermediate state which includes 
                one Rydberg atom and one ground state atom. 
        """
        UNmatdimension  = len(UNmat.toarray())
        #print(UNmat.toarray())
        n0 = self.n
        j0 = self.j
        l0 = self.l
        m0 = self.m1
        state_main = [self.nn,self.ll,self.jj,self.m2]
        d0 = self.atom1.getDipoleMatrixElement(n0,l0,j0,m0,self.nn,self.ll,self.jj,self.m2,0)
        Omega_array = []
        Omg0 = self.Omega0 
        for i in range(UNmatdimension):
            d = 0 
            if (self.basisStates[i][:4] == state_main) or (self.basisStates[i][4:] == state_main):
                d = self.atom1.getDipoleMatrixElement(n0,l0,j0,m0,self.nn,self.ll,self.jj,self.m2,0)
            Omega_array = np.append(Omega_array,0.5*Omg0*d/d0)
        row = np.zeros(UNmatdimension) + 1
        col = np.arange(UNmatdimension)
        mat = csr_matrix((Omega_array, (row, col)), shape=(2, UNmatdimension))
        UNmat = vstack([mat,UNmat])
        row = np.arange(UNmatdimension+2)
        row = np.concatenate((np.array([1]), row))
        col = np.zeros(UNmatdimension+2) + 1
        col = np.concatenate((np.array([0]), col))
        Omega_array = np.concatenate((np.array([Omg0*0.5,Omg0*0.5,self.Delta0]), Omega_array))
        mat = csr_matrix((Omega_array, (row, col)), shape=(UNmatdimension+2, 2))
        UNmat = hstack([mat,UNmat])
        UNmat = csr_matrix(UNmat)
        #print(UNmat.toarray())
        return UNmat
    
    def diagonalise(self, rangeR, noOfEigenvectors,
                    drivingFromState=[0, 0, 0, 0, 0],
                    eigenstateDetuning=0.,
                    sortEigenvectors=False,
                    progressOutput=False,
                    debugOutput=False):
        r"""
            Finds eigenstates.

            ARPACK ( :obj:`scipy.sparse.linalg.eigsh`) calculation of the
            `noOfEigenvectors` eigenvectors closest to the original state. If
            `drivingFromState` is specified as `[n,l,j,mj,q]` coupling between
            the pair-states and the situation where one of the atoms in the
            pair state basis is in :math:`|n,l,j,m_j\rangle` state due to
            driving with a laser field that drives :math:`q` transition
            (+1,0,-1 for :math:`\sigma^-`, :math:`\pi` and :math:`\sigma^+`
            transitions respectively) is calculated and marked by the
            colourmaping these values on the obtained eigenvectors.

            Args:
                rangeR ( :obj:`array`): Array of values for distance between
                    the atoms (in :math:`\mu` m) for which we want to calculate
                    eigenstates.
                noOfEigenvectors (int): number of eigen vectors closest to the
                    energy of the original (unperturbed) pair state. Has to be
                    smaller then the total number of states.
                eigenstateDetuning (float, optional): Default is 0. This
                    specifies detuning from the initial pair-state (in Hz)
                    around which we want to find `noOfEigenvectors`
                    eigenvectors. This is useful when looking only for couple
                    of off-resonant features.
                drivingFromState ([int,int,float,float,int]): Optional. State
                    of one of the atoms from the original pair-state basis
                    from which we try to drive to the excited pair-basis
                    manifold, **assuming that the first of the two atoms is
                    already excited to the specified Rydberg state**.
                    By default, program will calculate just
                    contribution of the original pair-state in the eigenstates
                    obtained by diagonalization, and will highlight it's
                    admixure by colour mapping the obtained eigenstates plot.
                    State is specified as :math:`[n,\ell,j,mj, d]`
                    where :math:`d` is +1, 0 or
                    -1 for driving :math:`\sigma^-` , :math:`\pi`
                    and :math:`\sigma^+` transitions respectively.
                sortEigenvectors(bool): optional, False by default. Tries to
                    sort eigenvectors so that given eigen vector index
                    corresponds to adiabatically changing eigenstate, as
                    detirmined by maximising overlap between old and new
                    eigenvectors.
                progressOutput (bool): optional, False by default. If true,
                    prints information about the progress of the calculation.
                debugOutput (bool): optional, False by default. If true,
                    similarly to progressOutput=True, this will print
                    information about the progress of calculations, but with
                    more verbose output.
        """

        self.r = np.sort(rangeR)
        dimension = len(self.basisStates)

        self.noOfEigenvectors = noOfEigenvectors

        # energy of the state - to be calculated
        self.y = []
        # how much original state is contained in this eigenvector
        self.highlight = []
        # what are the dominant contributing states?
        self.composition = []

        if (noOfEigenvectors >= dimension - 1):
            noOfEigenvectors = dimension - 1
            print("Warning: Requested number of eigenvectors >=dimension-1\n \
                 ARPACK can only find up to dimension-1 eigenvectors, where\
                dimension is matrix dimension.\n")
            if noOfEigenvectors < 1:
                return

        coupling = []
        self.maxCoupling = 0.
        self.maxCoupledStateIndex = 0
        if (drivingFromState[0] != 0):
            self.drivingFromState = drivingFromState
            if progressOutput:
                print("Finding coupling strengths")
            # get first what was the state we are calculating coupling with
            state1 = drivingFromState
            n1 = int(round(state1[0]))
            l1 = int(round(state1[1]))
            j1 = state1[2]
            m1 = state1[3]
            q = state1[4]

            for i in xrange(dimension):
                thisCoupling = 0.

                if (int(abs(self.basisStates[i][5] - l1)) == 1
                    and abs(self.basisStates[i][0]
                            - self.basisStates[self.originalPairStateIndex][0])
                        < 0.1
                    and abs(self.basisStates[i][1]
                            - self.basisStates[self.originalPairStateIndex][1])
                        < 0.1
                    and abs(self.basisStates[i][2]
                            - self.basisStates[self.originalPairStateIndex][2])
                        < 0.1
                    and abs(self.basisStates[i][3]
                            - self.basisStates[self.originalPairStateIndex][3])
                    < 0.1
                        ):
                    state2 = self.basisStates[i]
                    n2 = int(state2[0 + 4])
                    l2 = int(state2[1 + 4])
                    j2 = state2[2 + 4]
                    m2 = state2[3 + 4]
                    if debugOutput:
                        print(n1, " ", l1, " ", j1, " ", m1, " ", n2,
                              " ", l2, " ", j2, " ", m2, " q=", q)
                        print(self.basisStates[i])
                    dme = self.atom2.getDipoleMatrixElement(n1, l1, j1, m1,
                                                           n2, l2, j2, m2,
                                                           q,  s=self.s2)
                    thisCoupling += dme

                thisCoupling = abs(thisCoupling)**2
                if thisCoupling > self.maxCoupling:
                    self.maxCoupling = thisCoupling
                    self.maxCoupledStateIndex = i
                if (thisCoupling > 0.000001) and debugOutput:
                    print("original pairstate index = ",
                          self.originalPairStateIndex)
                    print("this pairstate index = ", i)
                    print("state itself ", self.basisStates[i])
                    print("coupling = ", thisCoupling)
                coupling.append(thisCoupling)

            print("Maximal coupling from a state")
            print("is to a state ",
                  self.basisStates[self.maxCoupledStateIndex])
            print("is equal to %.3e a_0 e" % self.maxCoupling)

        if progressOutput:
            print("\n\nDiagonalizing interaction matrix...\n")

        rvalIndex = 0.
        previousEigenvectors = []

        for rval in self.r:
            if progressOutput:
                sys.stdout.write("\r%d%%" %
                                 (rvalIndex / len(self.r - 1) * 100.))
                sys.stdout.flush()
            rvalIndex += 1.

            # calculate interaction matrix
            
            m = (self.matDiagonal).toarray() 
            #print(m)
            m[m!=0] += 2*self.Delta0
            #print(m)
            m = csr_matrix(m)
            rX = (rval * 1.e-6)**3
            for matRX in self.matR:
                m = m + matRX / rX
                rX *= (rval * 1.e-6)
            #Get the dressed state basis.
            m = self.__getDressedMatrixElements(m)
            # uses ARPACK algorithm to find only noOfEigenvectors eigenvectors
            # sigma specifies center frequency (in GHz)
            ev, egvector = eigsh(
                m, noOfEigenvectors,
                sigma=eigenstateDetuning * 1.e-9,
                which='LM',
                tol=1E-8)

            if sortEigenvectors:
                # Find which eigenvectors overlap most with eigenvectors from
                # previous diagonalisatoin, in order to find "adiabatic"
                # continuation for the respective states

                if previousEigenvectors == []:
                    previousEigenvectors = np.copy(egvector)
                    previousEigenvalues = np.copy(ev)
                rowPicked = [False for i in range(len(ev))]
                columnPicked = [False for i in range(len(ev))]

                stateOverlap = np.zeros((len(ev), len(ev)))
                for i in range(len(ev)):
                    for j in range(len(ev)):
                        stateOverlap[i, j] = np.vdot(
                            egvector[:, i], previousEigenvectors[:, j])**2

                sortedOverlap = np.dstack(
                    np.unravel_index(
                        np.argsort(stateOverlap.ravel()),
                        (len(ev), len(ev)))
                    )[0]

                sortedEigenvaluesOrder = np.zeros(len(ev), dtype=np.int32)
                j = len(ev)**2 - 1
                for i in range(len(ev)):
                    while rowPicked[sortedOverlap[j, 0]] or \
                            columnPicked[sortedOverlap[j, 1]]:
                        j -= 1
                    rowPicked[sortedOverlap[j, 0]] = True
                    columnPicked[sortedOverlap[j, 1]] = True
                    sortedEigenvaluesOrder[sortedOverlap[j, 1]
                                           ] = sortedOverlap[j, 0]

                egvector = egvector[:, sortedEigenvaluesOrder]
                ev = ev[sortedEigenvaluesOrder]
                previousEigenvectors = np.copy(egvector)

            self.y.append(ev)

            

            sh = []
            comp = []
            for i in xrange(len(ev)):
                sumCoupledStates =  abs(egvector[0, i])**2                      
                #comp.append(self._stateComposition(egvector[:, i]))
                sh.append(sumCoupledStates)
            #print(sh)
            self.highlight.append(sh)
            self.composition.append(comp)

        # end of FOR loop over inter-atomic dinstaces

    def exportData(self, fileBase, exportFormat="csv"):
        """
            Exports PairStateInteractions calculation data.

            Only supported format (selected by default) is .csv in a
            human-readable form with a header that saves details of calculation.
            Function saves three files: 1) `filebase` _r.csv;
            2) `filebase` _energyLevels
            3) `filebase` _highlight

            For more details on the format, see header of the saved files.

            Args:
                filebase (string): filebase for the names of the saved files
                    without format extension. Add as a prefix a directory path
                    if necessary (e.g. saving outside the current working directory)
                exportFormat (string): optional. Format of the exported file. Currently
                    only .csv is supported but this can be extended in the future.
        """
        fmt = 'on %Y-%m-%d @ %H:%M:%S'
        ts = datetime.datetime.now().strftime(fmt)

        commonHeader = "Export from Alkali Rydberg Calculator (ARC) %s.\n" % ts
        commonHeader += ("\n *** Pair State interactions for %s %s m_j = %d/2 , %s %s m_j = %d/2 pair-state. ***\n\n" %
                         (self.atom1.elementName,
                          printStateString(self.n, self.l, self.j), int(
                              round(2. * self.m1)),
                          self.atom2.elementName,
                          printStateString(self.nn, self.ll, self.jj), int(round(2. * self.m2))))
        if (self.interactionsUpTo == 1):
            commonHeader += " - Pair-state interactions included up to dipole-dipole coupling.\n"
        elif (self.interactionsUpTo == 2):
            commonHeader += " - Pair-state interactions included up to quadrupole-quadrupole coupling.\n"
        commonHeader += (" - Pair-state interactions calculated for manifold with spin angular momentum s1 = %.1d s2 = %.1d .\n"
                         % (self.s1, self.s2))

        if hasattr(self, 'theta'):
            commonHeader += " - Atom orientation:\n"
            commonHeader += "      theta (polar angle) = %.5f x pi\n" % (
                self.theta / pi)
            commonHeader += "      phi (azimuthal angle) = %.5f x pi\n" % (
                self.phi / pi)
            commonHeader += " - Calculation basis includes:\n"
            commonHeader += "      States with principal quantum number in range [%d-%d]x[%d-%d],\n" %\
                            (self.n - self.nRange, self.n + self.nRange,
                             self.nn - self.nRange, self.nn + self.nRange)
            commonHeader += "      AND whose orbital angular momentum (l) is in range [%d-%d] (i.e. %s-%s),\n" %\
                            (0, self.lrange, printStateLetter(
                                0), printStateLetter(self.lrange))
            commonHeader += "      AND whose pair-state energy difference is at most %.3f GHz\n" %\
                (self.energyDelta / 1.e9)
            commonHeader += "      (energy difference is measured relative to original pair-state).\n"
        else:
            commonHeader += " ! Atom orientation and basis not yet set (this is set in defineBasis method).\n"

        if hasattr(self, "noOfEigenvectors"):
            commonHeader += " - Finding %d eigenvectors closest to the given pair-state\n" %\
                            self.noOfEigenvectors

            if self.drivingFromState[0] < 0.1:
                commonHeader += " - State highlighting based on the relative contribution \n" +\
                    "   of the original pair-state in the eigenstates obtained by diagonalization.\n"
            else:
                commonHeader += (" - State highlighting based on the relative driving strength \n" +
                                 "   to a given energy eigenstate (energy level) from state\n" +
                                 "   %s m_j =%d/2 with polarization q=%d.\n" %
                                 (printStateString(*self.drivingFromState[0:3]),
                                  int(round(2. * self.drivingFromState[3])),
                                     self.drivingFromState[4]))

        else:
            commonHeader += " ! Energy levels not yet found (this is done by calling diagonalise method).\n"

        if exportFormat == "csv":
            print("Exporting StarkMap calculation results as .csv ...")

            commonHeader += " - Export consists of three (3) files:\n"
            commonHeader += ("       1) %s,\n" %
                             (fileBase + "_r." + exportFormat))
            commonHeader += ("       2) %s,\n" %
                             (fileBase + "_energyLevels." + exportFormat))
            commonHeader += ("       3) %s.\n\n" %
                             (fileBase + "_highlight." + exportFormat))

            filename = fileBase + "_r." + exportFormat
            np.savetxt(filename,
                       self.r, fmt='%.18e', delimiter=', ',
                       newline='\n',
                       header=(commonHeader +
                               " - - - Interatomic distance, r (\mu m) - - -"),
                       comments='# ')
            print("   Interatomic distances (\mu m) saved in %s" % filename)

            filename = fileBase + "_energyLevels." + exportFormat
            headerDetails = " NOTE : Each row corresponds to eigenstates for a single specified interatomic distance"
            np.savetxt(filename,
                       self.y, fmt='%.18e', delimiter=', ',
                       newline='\n',
                       header=(commonHeader +
                               ' - - - Energy (GHz) - - -\n' + headerDetails),
                       comments='# ')
            print("   Lists of energies (in GHz relative to the original pair-state energy)" +
                  (" saved in %s" % filename))

            filename = fileBase + "_highlight." + exportFormat
            np.savetxt(filename,
                       self.highlight, fmt='%.18e', delimiter=', ',
                       newline='\n',
                       header=(
                           commonHeader + ' - - - Highlight value (rel.units) - - -\n' + headerDetails),
                       comments='# ')
            print("   Highlight values saved in %s" % filename)

            print("... data export finished!")
        else:
            raise ValueError("Unsupported export format (.%s)." % format)

    def _addState(self, n1, l1, j1, mj1, n2, l2, j2, mj2):
        stateString = ""
        if (abs(self.s1 - 0.5) < 0.1):
            # Alkali atom
            stateString += "|%s %d/2" %\
                (printStateStringLatex(n1, l1, j1, s=self.s1), int(2 * mj1))
        else:
            # divalent atoms
            stateString += "|%s %d" %\
                (printStateStringLatex(n1, l1, j1, s=self.s1), int(mj1))

        if (abs(self.s2 - 0.5) < 0.1):
            # Alkali atom
            stateString += ",%s %d/2\\rangle" %\
                (printStateStringLatex(n2, l2, j2, s=self.s2), int(2 * mj2))
        else:
            # divalent atom
            stateString += ",%s %d\\rangle" %\
                (printStateStringLatex(n2, l2, j2, s=self.s2), int(mj2))
        return stateString

    def plotLevelDiagram(self, highlightScale='linear', removeacstarkshift = True, hlim = [0.9,1]):
        """
            Plots pair state level diagram

            Call :obj:`showPlot` if you want to display a plot afterwards.

            Args:
                highlightColor (string): optional, specifies the colour used
                    for state highlighting
                highlightScale (string): optional, specifies scaling of
                    state highlighting. Default is 'linear'. Use 'log-2' or
                    'log-3' for logarithmic scale going down to 1e-2 and 1e-3
                    respectively. Logarithmic scale is useful for spotting
                    weakly admixed states.
        """
        rvb = matplotlib.cm.get_cmap(name='hsv')
        # rvb = LinearSegmentedColormap.from_list('mymap',
        #                                         ['0.9', highlightColor])

        if highlightScale == 'linear':
            cNorm = matplotlib.colors.Normalize(vmin=hlim[0], vmax=hlim[1])
        elif highlightScale == 'log-2':
            cNorm = matplotlib.colors.LogNorm(vmin=1e-2, vmax=1)
        elif highlightScale == 'log-3':
            cNorm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=1)
        else:
            raise ValueError("Only 'linear', 'log-2' and 'log-3' are valid "
                             "inputs for highlightScale")

        print(" Now we are plotting...")
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7.0, 4.0))

        self.y = np.array(self.y)
        self.highlight = np.array(self.highlight)

        colorfulX = []
        colorfulY = []
        colorfulState = []

        for i in xrange(len(self.r)):
            for j in xrange(len(self.y[i])):
                colorfulX.append(self.r[i])
                colorfulY.append(self.y[i][j])
                colorfulState.append(self.highlight[i][j])

        #colorfulState = np.array(colorfulState)
        #sortOrder = colorfulState.argsort(kind='heapsort')
        colorfulX = np.array(colorfulX)
        colorfulY = np.array(colorfulY)
        if removeacstarkshift:
            acstark = -(np.sqrt((self.Omega0**2+ self.Delta0**2))/2-np.abs(self.Delta0/2))
            colorfulY = colorfulY+np.sign(-self.Delta0)*acstark
        #colorfulX = colorfulX[sortOrder]
        #colorfulY = colorfulY[sortOrder]
        # colorfulState = colorfulState[sortOrder]
        colorfulY = 1e6*colorfulY
        self.ax.scatter(colorfulX, colorfulY, s=10, c=colorfulState, linewidth=0,
                         norm=cNorm, cmap=rvb, zorder=2, picker=5)
        #self.ax.scatter(colorfulX, colorfulY, s=10, linewidth=0,
                       # norm=cNorm, cmap=rvb, zorder=2, picker=5)
        cax = self.fig.add_axes([0.91, 0.1, 0.02, 0.8])
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=rvb, norm=cNorm)

        if (self.drivingFromState[0] == 0):
            # colouring is based on the contribution of the original pair state here
            label = ""
            if (abs(self.s1-0.5) < 0.1):
                # Alkali atom
                label += r"$|\langle %s m_j=%d/2 " % \
                             (printStateStringLatex(self.n, self.l, self.j),
                              int(round(2. * self.m1, 0)))
            else:
                # divalent atom
                label += r"$|\langle %s m_j=%d " % \
                             (printStateStringLatex(self.n, self.l, self.j,
                                                    s=self.s1),
                              int(round(self.m1, 0)))

            if (abs(self.s2-0.5) < 0.1):
                # Alkali atom
                label += r", %s m_j=%d/2 | \mu \rangle |^2$" % \
                             (printStateStringLatex(self.nn, self.ll, self.jj),
                              int(round(2. * self.m2, 0)))
            else:
                # divalent atom
                label += r", %s m_j=%d | \mu \rangle |^2$" % \
                             (printStateStringLatex(self.nn, self.ll, self.jj,
                                                    s=self.s2),
                              int(round(self.m2, 0)))

            cb.set_label(label)
        else:
            # colouring is based on the coupling to different states
            cb.set_label(r"$(\Omega_\mu/\Omega)^2$")

        self.ax.set_xlabel(r"Interatomic distance, $R$ ($\mu$m)")
        self.ax.set_ylabel(r"Pair-state relative energy, $\Delta E/h$ (kHz)")

    def savePlot(self, filename="PairStateInteractions.pdf"):
        """
            Saves plot made by :obj:`plotLevelDiagram`

            Args:
                filename (:obj:`str`, optional): file location where the plot
                    should be saved
        """
        if (self.fig != 0):
            self.fig.savefig(filename, bbox_inches='tight')
        else:
            print("Error while saving a plot: nothing is plotted yet")
        return 0

    def showPlot(self, interactive=True):
        """
            Shows level diagram printed by
            :obj:`PairStateInteractions.plotLevelDiagram`

            By default, it will output interactive plot, which means that
            clicking on the state will show the composition of the clicked
            state in original basis (dominant elements)

            Args:
                interactive (bool): optional, by default it is True. If true,
                    plotted graph will be interactive, i.e. users can click
                    on the state to identify the state composition

            Note:
                interactive=True has effect if the graphs are explored in usual
                matplotlib pop-up windows. It doesn't have effect on inline
                plots in Jupyter (IPython) notebooks.


        """
        if interactive:
            self.ax.set_title("Click on state to see state composition")
            self.clickedPoint = 0
            self.fig.canvas.draw()
            self.fig.canvas.mpl_connect('pick_event', self._onPick)

        plt.show()
        return 0

    def _onPick(self, event):
        if isinstance(event.artist, matplotlib.collections.PathCollection):
            x = event.mouseevent.xdata
            y = event.mouseevent.ydata

            i = np.searchsorted(self.r, x)
            if i == len(self.r):
                i -= 1
            if ((i > 0) and (abs(self.r[i - 1] - x) < abs(self.r[i] - x))):
                i -= 1

            j = 0
            for jj in xrange(len(self.y[i])):
                if (abs(self.y[i][jj] - y) < abs(self.y[i][j] - y)):
                    j = jj

            # now choose the most higlighted state in this area
            distance = abs(self.y[i][j] - y) * 1.5
            for jj in xrange(len(self.y[i])):
                if (abs(self.y[i][jj] - y) < distance and
                        (abs(self.highlight[i][jj]) > abs(self.highlight[i][j]))):
                    j = jj

            if (self.clickedPoint != 0):
                self.clickedPoint.remove()

            self.clickedPoint, = self.ax.plot([self.r[i]], [self.y[i][j]], "bs",
                                              linewidth=0, zorder=3)

            self.ax.set_title("State = " + self.composition[i][j] +
                              ("   Colourbar = %.2f" % self.highlight[i][j]), fontsize=11)

            event.canvas.draw()

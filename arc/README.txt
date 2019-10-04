Elizabeth Robertson 2019

This is where the main body of the code which comprises DARC exists. 

__init__.py - this is used to tell the package which file to load when 
			  the package is imported. 

web_functionality.py - this has been left alone by me, will probably 
				be ammended when we push DARC to the git and integrate
				onto the web interface.

wigner.py - contains the wigner functions used in calculating angular
			matrix elements among other things. 

alkali_atom_data.py-contains all the Alklai classes with their respective
				data. If you need to modify DATA then you should look here
				and in the data files in .arc-dta in your home directory 
				all classes are instances of AlkaliAtom. This was all collated
				by Nikola Sibalic 

alkali_atom_functions.py currently contains the delaration of the AlkaliAtom 
				class and all the functions associated with this class. This 
				is soon to be ameneded as we are going to implement a more 
				general class. This was largely written by Nikola Sibalic
				with ammendments made to most classes to include integer	
				spin quantum numbers. 

earth_alkali_atom_data.py - contains all the Alklai classes with their respective
				data. If you need to modify DATA then you should look here
				and in the data files in .arc-dta in your home directory 
				all classes are instances of DivalentAtom. This was all collated
				by Elizabeth Robertson


earth_alkali_atom_fuction.py- currently contains the delaration of the DivalentAtom 
				class and all the functions associated with this class. This 
				is soon to be ameneded as we are going to implement a more 
				general class.

calculations_atom_single.py - this contains all of the code for the StarkMaps
				class and the LevelPlot class. This has been amened to work for
				

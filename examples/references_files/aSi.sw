# Stillinger-Weber parameters for various elements and mixtures
# multiple entries can be added to this file, LAMMPS reads the ones it needs
# these entries are in LAMMPS "metal" units:
#   epsilon = eV; sigma = Angstroms
#   other quantities are unitless

# format of a single entry (one or more lines):
#   element 1, element 2, element 3, 
#   epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q, tol

# Here are the original parameters in metal units, for Silicon from:
#
# Stillinger and Weber,  Phys. Rev. B, v. 31, p. 5262, (1985)
#
# Parameters for 'dia' Si
#Si Si Si 2.1683  2.0951  1.80  21.0  1.20  -0.333333333333
#         7.049556277  0.6022245584  4.0  0.0 0.0
#
# Parameters for amorphous Si  with the modified SW potential
#(R. L. C. Vink, G. T. Barkema, W. F. van der Weg et N. Mousseau, A semi-empirical potential for amorphous silicon, J. Non-Cryst. Sol. 282, 248-255 (2001))
Si Si Si 1.64833  2.0951  1.80  31.5  1.20  -0.333333333333
         7.049556277  0.6022245584  4.0  0.0 0.0 

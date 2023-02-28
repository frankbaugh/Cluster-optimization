import numpy as np 
import pygmo as pg
import os

## Solutions: http://doye.chem.ox.ac.uk/jon/structures/LJ.html
## https://www.math.umd.edu/~mariakc/LJClusters.html
## pygmo is a wrapper for scipy_optimize which adds in additional functionality

class system:

    def __init__(self, potential, N):

        self.potential = potential
        self.N = N
        ## WLOG set (x1, y1, z1, x2, y2) = 0, then the decision vector
        ## has 5 less dimensions
        self.xsize = 3 * N - 5

    def get_bounds(self):

        width = 2
        ## fill arrays with minimum bounds
        low_bound = np.zeros(self.xsize,)
        high_bound = np.full((self.xsize,), width)

        return (low_bound, high_bound)


    def fitness(self, x):

        U = np.array([0.])

        ## add on the fixed positions to the decision vector
        r0 = np.array([0,0,0,0,0])
        tot_atoms = np.concatenate((r0, x), axis=0)

        ## Split decision vector into set of position vectors
        vects = np.split(tot_atoms, self.N)

        ## compute potential of the norm between all vectors: O(N^2)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_ij = np.subtract(vects[j], vects[i])
                r = np.sqrt(r_ij.dot(r_ij))
                U[0] += self.potential(r)

        ## pygmo requires fitness to return a vector
        return U


def LJ(r):

    return 4 * (pow(r, -12) - pow(r, -6))

def morse(r):

    strength = 1.
    exp_term = np.exp(-r + strength)

    return (1.- exp_term)*(1.-exp_term)

def XYZ(positions, N):
    
    Nstr = str(N)
    filename = "cluster" + Nstr + ".xyz"
    f = open(filename, "w")
    f.write(Nstr + "\n")
    f.write("This is a comment line \n")

    r0 = np.array([0,0,0, 0, 0])
    tot_atoms = np.concatenate((r0, positions), axis=0)

    posvects = np.split(tot_atoms, N)

    for i in range(len(posvects)):
        xyz = "H   "

        for j in range(3):
            xyz += str(posvects[i][j])
            if j != 2:
                xyz += "   "
            else:
                xyz += "\n"

        f.write(xyz)

    os.system("open " + filename)

def main():

    ## Select number of atoms
    while True:
        number = input("Enter number of atoms ")
        try:
            N = int(number)
            if N < 1: 
                print("N must be positive integer")
                continue
            break
        except ValueError:
            print("That's not an int!")     
    
    while True:
        potential = input("Enter 1 for Lennard-Jones potential, or 2 for Morse ")
        if not potential in ["1", "2"]:
            print("Invalid Selection")
            continue
        break
    
    ## Select potential
    if potential == "1":
        prob = pg.problem(system(LJ, N))
        units = "\u03B5"
    else:
        prob = pg.problem(system(morse, N))
        units = "D_e"

    ## Many algorithms; covariance matrix evolution strategy seems to work best and fastest
    ## Finds the D5h global minimum about half the time. Depending on desired run time this can be improved
    algo = pg.algorithm(pg.cmaes(gen=2000))
    pop = pg.population(prob, 1000)
    pop = algo.evolve(pop)

    print("Minimum energy " + str(pop.champion_f[0]) + " " + units) 

    XYZ(pop.champion_x, N)


main()








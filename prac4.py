
## Solutions: http://doye.chem.ox.ac.uk/jon/structures/LJ.html
## https://www.math.umd.edu/~mariakc/LJClusters.html
## pygmo is a wrapper for scipy_optimize which adds in additional functionality

class system:

    def __init__(self, potential, N):

        self.N = N
        self.potential = potential
        ## WLOG set (x1, y1, z1, x2, y2, ) = 0, then the decision vector
        ## has 5 less dimensions
        self.xsize = 3 * N - 6

    def get_bounds(self):

        width = 3
        ## fill arrays with minimum bounds
        low_bound = np.zeros(self.xsize)
        high_bound = np.full((self.xsize), width)

        return (low_bound, high_bound)

    def fitness(self, x):
        ## pygmo requires fitness to return a vector
        U = np.array([0.])

        ## add on the fixed positions to the decision vector
        r0 = np.array([0.,0.,0.,0.,0.])
        x = np.insert(x, 1, 0.)
        tot_atoms = np.concatenate((r0, x), axis=0)

        ## Split decision vector into set of position vectors
        vects = np.split(tot_atoms, self.N)

        ##This could really be alot more efficient but its fairly quick for N=7
        ## compute potential of the norm between all vectors: O(N^2)
        for i in range(self.N):
            for j in range(i+1, self.N):
                r_ij = np.subtract(vects[j], vects[i])
                r = np.linalg.norm(r_ij)
                U[0] += self.potential(r)

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

    ## Assemble positions from winning decision vector
    r0 = np.array([0.,0.,0., 0., 0.])
    positions = np.insert(positions, 1, 0.)
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

    ## Open the created file in the system's default .xyz program
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
    
    # Select inter-atomic potential
    while True:
        potential = input("Enter 1 for Lennard-Jones potential, or 2 for Morse ")
        if potential == "1":
            prob = pg.problem(system(LJ, N))
            units = "\u03B5"
            break
        elif potential == "2":
            prob = pg.problem(system(morse, N))
            units = "D_e"
            break
        else: continue
    
    ## Many minimization algorithms; covariance matrix evolution strategy seems to work best and fastest
    ## the gen, n and pop_size parameters want to be as low as possilbe (speed) whilst
    ## Still consistently hitting the global minimum D5h cluster

    algo = pg.algorithm(pg.cmaes(gen=500))

    # The real strength of pygmo is that it can compute in a parallelized fashion
    ## n 'islands' of evolving populations
    archi = pg.archipelago(n=8, algo=algo, prob=prob, pop_size=30, udi=pg.mp_island(use_pool=True))
    archi.evolve()
    print(archi)
    archi.wait()

    ## Extract best individual from archipelago
    min_index = np.argmin(archi.get_champions_f())
    min_energy = archi.get_champions_f()[min_index]
    configuration = archi.get_champions_x()[min_index]
    
    print("Minimum energy " + str(min_energy) + " " + units) 

    XYZ(configuration, N)

if __name__ == '__main__':       
    import numpy as np 
    import pygmo as pg
    import os
    main()

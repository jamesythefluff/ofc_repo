import numpy as np
import matplotlib.pyplot as plt


class Parameters:

    def __init__(self):
        self.L = 32
        self.Fth = 1.0
        self.K = 1.0
        self.KL = 1.0
        self.iterations = 10000
        self.begin_data = 6000
        self.a = self.K/(4*self.K + self.KL)
        self.ae = self.K/(3*self.K + self.KL)
        self.ac = self.K/(2*self.K + self.KL)



def drive(F, stack, params = Parameters()):
    Fth = params.Fth
    Fmax = np.max(F)
    position = np.argwhere(F == Fmax)[0]
    stack.add(tuple(position)) 
    Fadd = Fth - Fmax
    F += Fadd
    


def relax_stack(F, stack, params = Parameters()):
    
    new_stack = set()
    a = params.a
    ae = params.ae
    ac = params.ac
    Fth = params.Fth
    L = params.L
    
    for coord in stack:
        i = coord[0]
        j = coord[1]
        F_ij = F[i,j]
        F[i,j] = 0 # Reset the force at the site that has slipped
        
        N = [-1,1] # Used for iterating through nearest neighbours
        
        # Transfer force to nearest neighbours
        # Need to deal with 4 corners, 4 edges and main body separately due to different force transfer ratios
        if i == 0:

            if j == 0: 
                # (0,0) corner case
                F[i+1,j] += ac*F_ij
                F[i,j+1] += ac*F_ij

                if F[i+1,j] >= Fth:
                    new_stack.add((i+1,j))

                if F[i,j+1] >= Fth:
                    new_stack.add((i,j+1))

            elif j == L:
                # (0,L) corner case
                F[i+1,j] += ac*F_ij
                F[i,j-1] += ac*F_ij

                if F[i+1,j] >= Fth:
                    new_stack.add((i+1,j))

                if F[i,j-1] >= Fth:
                    new_stack.add((i,j-1))

            else:
                # (0,y) edge case
                F[i+1,j] += ae*F_ij
                F[i,j+1] += ae*F_ij
                F[i,j-1] += ae*F_ij

                if F[i+1,j] >= Fth:
                    new_stack.add((i+1,j))

                if F[i,j+1] >= Fth:
                    new_stack.add((i,j+1))

                if F[i,j-1] >= Fth:
                    new_stack.add((i,j-1))
                      
        elif i == L:
            if j == 0:
                # (L,0) corner case
                F[i-1,j] += ac*F_ij
                F[i,j+1] += ac*F_ij

                if F[i-1,j] >= Fth:
                    new_stack.add((i-1,j))

                if F[i,j+1] >= Fth:
                    new_stack.add((i,j+1))  

            elif j == L:
                # (L,L) corner case
                F[i-1,j] += ac*F_ij
                F[i,j-1] += ac*F_ij

                if F[i-1,j] >= Fth:
                    new_stack.add((i-1,j))

                if F[i,j-1] >= Fth:
                    new_stack.add((i,j-1))
            else:
                # (L,y) edge case
                F[i-1,j] += ae*F_ij
                F[i,j+1] += ae*F_ij
                F[i,j-1] += ae*F_ij

                if F[i-1,j] >= Fth:
                    new_stack.add((i-1,j))

                if F[i,j+1] >= Fth:
                    new_stack.add((i,j+1))

                if F[i,j-1] >= Fth:
                    new_stack.add((i,j-1))  

        else:
            if j == 0:
                # (x,0) edge case
                F[i+1,j] += ae*F_ij
                F[i-1,j] += ae*F_ij
                F[i,j+1] += ae*F_ij

                if F[i+1,j] >= Fth:
                    new_stack.add((i+1,j))

                if F[i-1,j] >= Fth:
                    new_stack.add((i-1,j))

                if F[i,j+1] >= Fth:
                    new_stack.add((i,j+1))

            elif j == L:
                # (x,L) edge case
                F[i+1,j] += ae*F_ij
                F[i-1,j] += ae*F_ij
                F[i,j-1] += ae*F_ij

                if F[i+1,j] >= Fth:
                    new_stack.add((i+1,j))

                if F[i-1,j] >= Fth:
                    new_stack.add((i-1,j))

                if F[i,j-1] >= Fth:
                    new_stack.add((i,j-1)) 

            else:
                # (x,y) main body case
                for n in N:
                    F[i+n,j] += a*F_ij
                    F[i,j+n] += a*F_ij

                    if F[i+n,j] >= Fth:
                        new_stack.add((i+n,j))

                    if F[i,j+n] >= Fth:
                        new_stack.add((i,j+n))
                     
    return F, new_stack

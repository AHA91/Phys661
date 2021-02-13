##### Meri Khurshudyan
##### Assignment 1
import numpy as np

#################### Remove comments to print ####################
#################### Part 1 Mag of a vector ######################

v = [1,2,3,4,5]
Mag_v = 0
for i in range(len(v)):
    Mag_v+=v[i]**2
Mag_v = np.sqrt(Mag_v)

#print(Mag_v)

    
#################### Part 2 Scalar Product #######################

a = [1,2,3,4,5]
b = [6,7,8,9,10]
Scal_prod = 0
for j in range(len(a)):
    Scal_prod += a[j]*b[j]

#print(Scal_prod)

####################### Matrix Algebra ###########################
####################### Using For Loops ##########################

####################### Part I ###################################

x = [1,2,3,4,5] #5x1 rows and columns behave the same in python we
                #can simply modify the for loop to behave as we wish

A = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]] #3x5
sum_a = 0
result = []
for k in range(len(A)):
    for m in range(len(A[1])):
        sum_a += A[k][m]*x[m]
    result.append(sum_a)
    sum_a = 0

#print(result)

###################### Part II ##################################
# I take p = 4 where A = mxp and B = pxn
_A_ = [[1,2,3,4],[5,6,7,8],[9,10,11,12]] # A = 3x4
_B_ = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] # B = 4x4
total = 0
resultant = [[None for x in range(len(_B_))] for y in range(len(_A_))] 
for g in range(len(_A_)):
    for t in range(len(_A_[1])): # delays the first for loop to remain on same row
        for f in range(len(_B_)):
            total += _A_[g][f]*_B_[f][t]
        resultant[g][t] = total
        total = 0

#print(resultant)


















    
    
        
    

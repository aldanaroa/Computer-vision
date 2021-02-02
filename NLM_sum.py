#Find Sum in Volume of Blocks
#contains 5 functions: NL_sum, LocalMeanVar, FindWeights, FindBlockValue, MatrixBlocks
import numpy as np
from IPython.core.debugger import set_trace
from IPython.core.debugger import Pdb


def NLM_sum(MR, alpha, n, M, u1, sigma1, sigma_est, beta):
#Within the Volume, calculate weights and filter
    [x,y,z] = MR.shape
    Vol_M = np.zeros((x,y,z))
    [UB_mean, UB_var] = LocalMeanVar(MR, alpha, n)
    #matrix with values of voxels per Block
    n_blocks = (2*M+1)**3
    Block_v = np.zeros(( n_blocks, (2*alpha+1)**3))
    #Matrix with linear indexes of MR. The volume to be filtered.
    Volume_ind = np.arange(x*y*z)
    Volume_ind = Volume_ind.reshape(x,y,z)
    #Matrix with voxels intensities per block. Columns are linear indexes
    Volume_voxels=np.zeros(((n_blocks, x*y*z)))
    #Mat_Blocks is matrix with ACTUAL values of voxels per block
    Mat_Blocks = MatrixBlocks(MR, alpha, n, n_blocks)
    #Loop through the blocks of Volume
    block_number = 0
    for i in range(n-1, x, n):
        for j in range(n-1, y, n):
            for k in range(n-1, z, n):
                #Bik is a block of shape (2xalpha +1)^3
                Bik = MR[i-alpha:i+alpha+1,j-alpha:j+alpha+1,k-alpha:k+alpha+1]
                mean_Bik = UB_mean[int(i/n),int(j/n),int(k/n)] 
                var_Bik = UB_var[int(i/n),int(j/n),int(k/n)]
                #Weights is of size [(2M+1) x  (2M+1) x (2M+1)]
                Weights = FindWeights(mean_Bik, var_Bik, UB_mean, UB_var, n, M, u1,sigma1)
                #Block_v is of size [(2M+1)^3 x  (2alpha+1)^3]               
                Block_v[block_number,:] = FindBlockValue(Bik, Weights, Mat_Blocks, alpha,n,M,sigma_est, beta)
                #Assign intensities in Volume_voxels columns corresponding to linear indexes of voxels in Block. 
                #This is used to find overlapping Block voxels           
                tmp=Volume_ind[i-alpha:i+alpha+1,j-alpha:j+alpha+1,k-alpha:k+alpha+1]
                tmp2=Block_v[block_number,:]
                Volume_voxels[block_number,tmp.flatten()]= tmp2.flatten()
                block_number = block_number +1
            
            
    #Find final Volume intensities according to Volume_voxels matrix. 
    #Column average

    Volume_ave=np.zeros((1, x*y*z))
    #Vector with sum of repeated voxels per linear index
    Volume_over=np.ones((1, x*y*z))   
    
    #set_trace()
    #number of non-zero values for each linear index
    Volume_over = np.sum(Volume_voxels > 0, 0)
    Volume_over[Volume_over==0] = 1
    #Sum of intensities for the same linear index
    Volume_ave = np.sum(Volume_voxels, 0)   
    
    #Average value for each linear index
    Volume_ave = Volume_ave/Volume_over
    
    #Assign Final Result to Vol_M. Need to map linear indexes to cube   
    Volume_ind = np.arange(x*y*z)
    Vol_M = Vol_M.flatten() #added because change from Matlab to python
    Vol_M[Volume_ind] = Volume_ave
    Vol_M = Vol_M.reshape(x,y,z)
    
    return Vol_M

def LocalMeanVar(MR, alpha, n):

    if 2*alpha >= n :
        [x,y,z] = MR.shape        
        UB_mean = np.zeros((x,y,z))
        UB_var = np.zeros((x,y,z))
        for i in range(n-1,x,n):        
            for j in range(n-1,y,n):        
                for k in range (n-1,z,n): 
                    tmp = MR[i-alpha:i+alpha+1,j-alpha:j+alpha+1,k-alpha:k+alpha+1]        
                    UB_mean[i,j,k]  = np.mean(tmp.flatten())       
                    UB_var[i,j,k] = np.var(tmp.flatten())
        
        UB_mean = UB_mean[n-alpha: :n,n-alpha: :n, n-alpha: :n]        
        UB_var    =     UB_var[n-alpha: :n,n-alpha: :n, n-alpha: :n]
    else:
        print('Error: remember 2*alpha >= n')
        UB_mean = 0
        UB_var = 0
    
    return UB_mean, UB_var

def FindWeights(mean_Bik, var_Bik, UB_mean, UB_var, n, M, u1,sigma1):

    #[x,y,z] = UB_mean.shape   
    #Weights = np.zeros((1, x*y*z))
    Vol_size = (2*M+1)**3    
    Val_mean = np.zeros((Vol_size,1))    
    Val_var = np.zeros((Vol_size,1))   
    #Within the search volume, calculate weights
    #Ratios to compare
    #Val_mean = mean_Bik/UB_mean.flatten()
    #convertir nan a 0. Agregado en Python
    #Val_mean = np.nan_to_num(Val_mean,nan=0)
    #Val_var = var_Bik/UB_var.flatten()
    #convertir nan a 0. Agregado en Python
    #Val_var = np.nan_to_num(Val_var,nan=0)
    #Fill matrix of weights with 1 if thresholds are met
    #Old line Matlab
    #Wei = (u1 < Val_mean)& (Val_mean < 1/u1)&(sigma1< Val_var)&(Val_var < 1/sigma1)
    #New line
    #set_trace()
    Wei = (u1* UB_mean.flatten()<= mean_Bik)& (mean_Bik <= UB_mean.flatten()/u1)&(sigma1*UB_var.flatten()<= var_Bik)&(var_Bik<= UB_var.flatten()/sigma1)
    Weights = Wei.reshape((1, Vol_size))
    #Weights = Wei
    return Weights


def  FindBlockValue(Bik,Weights, MB, alpha, n, M, sigma_est, beta):
    #Receive Bik Weights of Blocks and Volume
    #Calculate Block Value [ 1x 2alpha +1] vector
    #first simulation was with Beta=0.5 = 2xBetareal=2*1/4   
    Beta= beta
    #[x,y,z]= Weights.shape
    [x, length_w] = Weights.shape
    #length_w = x*y*z
    #length_w = x    
    block_size = (2*alpha +1)**3    
    shape_value = 1/2/block_size/sigma_est/Beta
    #Bik_mat is the matrix with Bik as rows
    #Bik_mat = np.tile(Bik.flatten().T, (length_w, 1))    
    #Vector with exponentials distance values    
    sum_v = np.zeros((1, length_w))
    #set_trace()
    distance = np.sum( (Bik.flatten() - MB)**2, axis=1, keepdims=True)   
    sum_v= np.exp(-(1/shape_value)*(distance.T))    
    tmp= Weights*sum_v  
    Zik=np.sum(tmp)
    Block_v = tmp@MB /Zik
    return Block_v

#nueva función en código python para no repetir operación de distancia con un cubo ya conocido de intensidades
def  MatrixBlocks(MR, alpha, n, n_blocks):
    #Receive MR to create matrix with rows equals to Blocks and columns equal to intensities
    #Mat_Block matrix of [(2M+1)³ x 2alpha +1]
    [xv,yv,zv] = MR.shape
    block_size = (2*alpha +1)**3    
    Mat_Blocks = np.zeros((n_blocks, block_size))
    block_number=0
    for i in range(n-1,xv,n):
        for j in range(n-1,yv,n):
            for k in range(n-1,zv,n):
                Mat_Blocks[block_number, :] = MR[i-alpha:i+alpha+1,j-alpha:j+alpha+1,k-alpha:k+alpha+1].flatten()              
                block_number+=1
                
    return Mat_Blocks  


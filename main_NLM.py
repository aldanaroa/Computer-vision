def main_NLM(MR,alpha, n, M, u1,sigma1,sigma_est, beta):
    import numpy as np
    import NLM_sum as nls
    import importlib
    from IPython.core.debugger import set_trace
    
    importlib.reload(nls)
#MR = cubo inicial
#n = espacio entre centros
#M = Volumen de búsqueda
#alpha, define tamaño de bloque |Bi| = (2alpha +1)^3
#u1 = 0.95
#sigma1 = sigma1² = 0.5
#sigma_est = noisevar(MR) estimación de sigma
#beta = parameter to be tuned 
    [xn,yn,zn] = MR.shape
    #index for original cube
    ind = n*(2*M+1)
    Filtered_image= np.zeros((xn,yn,zn))

#within image, divide cube, sent to for 

    ix=xn-ind
    iy= yn-ind
    iz=zn-ind       
    
    for i in range(0, ix, ind): 
       for j in range(0, iy, ind):
           for k in range(0, iz, ind):
               # Send Search Volume whose size is ind
                Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1]=nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)                

#Calculate faces, laterals and corner not included in first sweep
#face 3 
    for i in range(ix-1, ix, ind):      
        for j in range(0, iy, ind):    
            for k in range(0, iz, ind):
                #Send Search Volume whose size is ind       
                Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1]=nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
#face 2
    for j in range(iy-1, iy, ind):
        for i in range(0 , ix, ind):
            for k in range(0, iz, ind):
#               Send Search Volume whose size is ind
               Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1] = nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
 
#face 1                
    for k in range(iz-1, iz, ind):
        for i in range(0 , ix, ind):
           for j in range(0, iy, ind):
#              Send Search Volume whose size is ind        
             Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1] = nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
                 
#lateral 1
    for i in range(ix-1, ix, ind):     
        #set_trace()
        for j in range(iy-1, iy, ind):    
            for k in range(0, iz, ind):
               #Send Search Volume whose size is ind       
               Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1] = nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
               #print(ix, iy, iz)
    
#lateral 2
    for i in range(ix-1, ix, ind):     
        #set_trace()
        for j in range(0,iy, ind):    
            for k in range(iz-1, iz, ind):
               #Send Search Volume whose size is ind       
               Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1] = nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
               #print(ix, iy, iz)

#lateral 3
    for i in range(0, ix, ind):     
        #set_trace()
        for j in range(iy-1,iy, ind):    
            for k in range(iz-1, iz, ind):
               #Send Search Volume whose size is ind       
               Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1] = nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
               #print(ix, iy, iz)

#corner
    for i in range(ix-1, ix, ind):     
        #set_trace()
        for j in range(iy-1,iy, ind):    
            for k in range(iz-1, iz, ind):
               #Send Search Volume whose size is ind       
               Filtered_image[i:i+ind+1, j:j+ind+1, k:k+ind+1] = nls.NLM_sum(MR[i:i+ind+1, j:j+ind+1, k:k+ind+1], alpha, n, M, u1,sigma1,sigma_est, beta)
               #print(ix, iy, iz)



                              
    return Filtered_image    


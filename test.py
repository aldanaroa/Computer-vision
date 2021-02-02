import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
#import main_NLM as mln
#import importlib
#    
#importlib.reload(mln)

path='./02/3D/MR.mnc'
img1 = nib.load('./02/2d.mnc')
img2 = nib.load(path)
img2_data = img2.get_fdata()
img2_data.shape
img2_data = img2_data/img2_data.max()
type(img2_data)
img2_data.max()
def show_slices(slices):
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
   return
 
slice_0 = img2_data[26, :, :]
slice_1 = img2_data[:, 30, :]
slice_2 = img2_data[:, :, 32]

show_slices([slice_0, slice_1, slice_2])
plt.suptitle("xyz plane slices for a brain")  

#%run main_NLM.py
#%run NLM_sum.py
#v=0.029854006995932925

#M3=mln.main_NLM(img2_data, 1, 2, 5, 0.95,0.5,v)
#slice_0 = M3[26, :, :]
#slice_1 = M3[:, 30, :]
#slice_2 = M3[:, :, 32]
#
#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for image") 

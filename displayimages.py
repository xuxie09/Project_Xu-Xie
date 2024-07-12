#from extractslices import extractslices
#from extractpatches import extractpatches
import numpy as np
import scipy.spatial.distance as ssd 
from PIL import Image
#from scipy.misc import toimage
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import cv2
from hausdorff import hausdorff_distance


def displayimages(pred,orig,subject_no,slice_no,pred_csf,pred_gm,pred_wm,orig_csf,orig_gm,orig_wm,image_test_array):
    
    
    def combine_patches(images, patch):
        patch1, patch2, patch3, patch4 = images[patch], images[patch+1], images[patch+2], images[patch+3]
        top_row = np.concatenate((patch1, patch2), axis=1)  # 横向拼接上方两块
        bottom_row = np.concatenate((patch3, patch4), axis=1)  # 横向拼接下方两块
        full_image = np.concatenate((top_row, bottom_row), axis=0)  # 纵向拼接两行
        return full_image  

         
    patch = subject_no*slice_no 
    #patch = 340
    Combine_Pred = combine_patches(pred, patch)
    Final_Pred = (Combine_Pred[:,:,1]*1+Combine_Pred[:,:,2]*2+Combine_Pred[:,:,3]*3)
    Combine_Orig = combine_patches(orig, patch)
    Final_Orig = (Combine_Orig[:,:,1]*1+Combine_Orig[:,:,2]*2+Combine_Orig[:,:,3]*3)
    # Final_Pred = (pred[patch,:,:,1]*1)+(pred[patch,:,:,2]*2)+(pred[patch,:,:,3]*3)
    # Final_Orig = (orig[patch,:,:,1]*1)+(orig[patch,:,:,2]*2)+(orig[patch,:,:,3]*3)
    
    #Inter_GT = image_test_array[patch,:,:]
    Inter_GT = combine_patches(image_test_array, patch)
    d1,d2,d3 = Inter_GT.shape
    Final_GT = Inter_GT.reshape(d1,d2*d3)
    
    

    
    # Final_csf = (pred_csf[patch,:,:])
    # Final_gm = (pred_gm[patch,:,:])
    # Final_wm = (pred_wm[patch,:,:])
    # Final_orig_csf = (orig_csf[patch,:,:])
    # Final_orig_gm = (orig_gm[patch,:,:])
    # Final_orig_wm = (orig_wm[patch,:,:])
    Final_csf = combine_patches(pred_csf, patch)
    Final_gm = combine_patches(pred_gm,patch)
    Final_wm = combine_patches(pred_wm,patch)
    Final_orig_csf = combine_patches(orig_csf,patch)
    Final_orig_gm = combine_patches(orig_gm,patch)
    Final_orig_wm = combine_patches(orig_wm,patch)
    
    
    np.random.seed(0)
    print("Hausdorff distance CSF: {0}".format( hausdorff_distance(Final_csf, Final_orig_csf, distance="euclidean") ))
    print("Hausdorff distance GM: {0}".format( hausdorff_distance(Final_gm,Final_orig_gm, distance="euclidean") ))
    print("Hausdorff distance WM: {0}".format( hausdorff_distance(Final_wm, Final_orig_wm, distance="euclidean") ))
         
    plt.figure(figsize=(15, 10))
    #plt.figure(figsize=(5, 5))
    plt.subplot(231)
    plt.imshow(Final_GT, cmap = 'gray', interpolation ='bicubic')    
    plt.title('Img')
    #plt.show()

    #plt.figure(figsize=(5, 5))
    plt.subplot(232)
    plt.imshow(Final_Orig, cmap = 'gray', interpolation ='bicubic')
    plt.title('Orig')
    #plt.show()
    
    #plt.figure(figsize=(5, 5))
    plt.subplot(233)
    plt.imshow(Final_Pred, cmap = 'gray', interpolation ='bicubic')
    plt.title('Pred')
    #plt.show()
    
    #plt.figure(figsize=(5, 5))
    plt.subplot(234)
    plt.imshow(Final_csf, cmap = 'gray', interpolation ='bicubic')    
    plt.title('CSF')
    #plt.show()

    #plt.figure(figsize=(5, 5))
    plt.subplot(235)
    plt.imshow(Final_gm, cmap = 'gray', interpolation ='bicubic')
    plt.title('GM')
    #plt.show()

    #plt.figure(figsize=(5, 5))
    plt.subplot(236)
    plt.imshow(Final_wm, cmap = 'gray', interpolation ='bicubic')
    plt.title('WM')
    
    plt.show()

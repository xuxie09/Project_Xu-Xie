import numpy as np
def extractpatches (trn_imagedatas, trn_maskdatas, tst_imagedatas, tst_maskdatas, windowsize_r,windowsize_c,num_patch):

    def creatpatches(image,windowsize_r,windowsize_c,num_patch):
        x = 0
        window = np.ndarray([num_patch,windowsize_r,windowsize_c])
        for r in range(0,image.shape[0], windowsize_r):    
            for c in range(0,image.shape[1], windowsize_c):
                window[x] = image[r:r+windowsize_r,c:c+windowsize_c]
                x = x + 1
        return window
    
    def creatpatches3D(image,windowsize_r,windowsize_c,num_patch):
        x = 0
        window = np.ndarray([num_patch,windowsize_r,windowsize_c,4])
        for r in range(0,image.shape[0], windowsize_r):    
            for c in range(0,image.shape[1], windowsize_c):
                window[x] = image[r:r+windowsize_r,c:c+windowsize_c,:]
                x = x + 1
        return window
    
   
    
    
  
    train_img = []
    train_mask = []
    for i in range(len(trn_imagedatas)):
        image = trn_imagedatas[i]
        mask = trn_maskdatas[i]
        window_img = creatpatches(image.reshape(image.shape[0],image.shape[1]),windowsize_r,windowsize_c,num_patch)
        window_msk = creatpatches3D(mask.reshape(mask.shape[0],mask.shape[1],4),windowsize_r,windowsize_c,num_patch)
        train_img.append(window_img)
        train_mask.append(window_msk)
        
    train_img = np.array(train_img)
    train_mask = np.array(train_mask)
    
    train_img = train_img.reshape(train_img.shape[0]*train_img.shape[1], train_img.shape[2], train_img.shape[3], 1)
    train_mask = train_mask.reshape(train_mask.shape[0]*train_mask.shape[1], train_mask.shape[2], train_mask.shape[3], 4)
    
    test_img = []
    test_mask = []
    
    for i in range(len(tst_imagedatas)):
        image = tst_imagedatas[i]
        mask = tst_maskdatas[i]
        
        window_img = creatpatches(image.reshape(image.shape[0],image.shape[1]),windowsize_r,windowsize_c,num_patch)
        window_msk = creatpatches3D(mask.reshape(mask.shape[0],mask.shape[1],4),windowsize_r,windowsize_c,num_patch)
        test_img.append(window_img)
        test_mask.append(window_msk)
        
    test_img = np.array(test_img)
    test_mask = np.array(test_mask)



    test_img = test_img.reshape(test_img.shape[0]*test_img.shape[1], test_img.shape[2], test_img.shape[3], 1)
    test_mask = test_mask.reshape(test_mask.shape[0]*test_mask.shape[1], test_mask.shape[2], test_mask.shape[3], 4)
    return (train_img,train_mask,test_img,test_mask)

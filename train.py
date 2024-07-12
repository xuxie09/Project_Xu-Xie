from extractslices import extractslices

from Unet import Attention_U_Net
#from Unet import Unet
from keras.callbacks import EarlyStopping
from displayresults import displayresults
from displayimages import displayimages
import matplotlib.pyplot as plt
from extractpatches import extractpatches
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import scipy.spatial.distance as ssd 

imgs_train_array = np.load(r'/home/hpc/iwbi/iwbi105h/seg/Org_train_data.npy')
mask_train_array = np.load(r'/home/hpc/iwbi/iwbi105h/seg/GT_train_data.npy')
image_test_array = np.load(r'/home/hpc/iwbi/iwbi105h/seg/Org_test_data.npy')
mask_test_array = np.load(r'/home/hpc/iwbi/iwbi105h/seg/GT_test_data.npy')

windowsize_r = 128
windowsize_c = 128
num_patch = 4
num_channels = 1
output_channels = 4
#(imgs_train_array, mask_train_array,image_test_array, mask_test_array) = extractpatches(imgs_train_array, mask_train_array,image_test_array, mask_test_array,windowsize_r,windowsize_c)
(imgs_train, mask_train, image_test, mask_test) = extractpatches(imgs_train_array, mask_train_array, image_test_array, mask_test_array, windowsize_r, windowsize_c, num_patch)
model = Attention_U_Net(windowsize_r,windowsize_c,num_channels,output_channels)
#model = Unet(windowsize_r,windowsize_c)
# optimizer.step()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.0001, restore_best_weights=True)
history = model.fit(imgs_train, mask_train, batch_size=1, epochs=10, verbose=1, validation_split=0.2, shuffle=True,callbacks=[es])

masks_pred = model.predict(image_test, batch_size=1, verbose=1)
np.save('mask_pred.npy', masks_pred)
gm_dsc, gm_jc, wm_dsc, wm_jc,csf_dsc, csf_jc,pred_csf,pred_gm,pred_wm,orig_csf,orig_gm,orig_wm = displayresults(masks_pred,mask_test)

print("Gray Matter:",gm_dsc, gm_jc)
print("White Matter:",wm_dsc, wm_jc)
print("CSF:",csf_dsc, csf_jc)

#subject_no = 1
#slice_no = 340
#displayimages(masks_pred,mask_test_array,subject_no,slice_no,pred_csf,pred_gm,pred_wm,orig_csf,orig_gm,orig_wm,image_test_array,"output_image.png")
MSE = np.square(np.subtract(mask_test, masks_pred)).mean()
print("MSE:",MSE)
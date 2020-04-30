
# coding: utf-8

# In[28]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import io, transform

import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[36]:


transform = transforms.Compose(
    [
     transforms.RandomAffine(0, translate=(0.2,0.2), scale=None, shear=None, resample=False, fillcolor=0),# randomly translate:-img_width * 0.2 < dx < img_width * 0.2
     transforms.ToTensor(), # convert Python Image Library (PIL) format to PyTorch tensors.
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # convert the data from [0,1] to [-1,1]
    ]) 


testset = torchvision.datasets.CIFAR10(root='./data', 
                                       train=False,
                                       download=True, 
                                       transform= transform
                                       )

testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=5,
                                         shuffle=False)


# In[37]:


# unnormalize the images
def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1,2,0)


# In[38]:



# def shift_h(im, shift_amount):
#     """Shift the image horizontally by shift_amount pixels, 
#     use positive numbers to shift the image to the right,
#     use negative numbers to shift the image to the left"""
#     if shift_amount == 0:
#         return im
#     else:
#         if len(im.shape) == 3: # for a single image
#             new_image = torch.zeros_like(im)
#             new_image[:, :, :shift_amount] = im[:,:,-shift_amount:]
#             new_image[:,:,shift_amount:] = im[:,:,:-shift_amount]
#             return new_image
#         elif len(im.shape) == 4: # for batches of images
#             new_image = torch.zeros_like(im)
#             new_image[:, :, :, :shift_amount] = im[:, :, :, -shift_amount:]
#             new_image[:, :, :, shift_amount:] = im[:, :, :, :-shift_amount]
#             return new_image


dataiter = iter(testloader)
images, labels = dataiter.next()

fig, axes = plt.subplots(1, len(images), figsize=(12,2.5))
for idx, image in enumerate(images):
    image = convert_to_imshow_format(image)
#     image = shift_h(image,5)
    axes[idx].imshow(image)
#     axes[idx].set_title(classes[labels[idx]])


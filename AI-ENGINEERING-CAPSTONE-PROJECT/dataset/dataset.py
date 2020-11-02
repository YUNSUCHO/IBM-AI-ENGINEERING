# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os
torch.manual_seed(0)



# Create your own dataset object

class Data(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="/Users/yun/Desktop/OnlineCourse/Coursera/AI-ENGINEERING/Datas/" 
        positive="Positive_tensors"
        negative='Negative_tensors'

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        
        number_of_samples=len(positive_files)+len(negative_files)
        
        self.all_files=[None]*number_of_samples
        print(len(self.all_files))
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        
        # The transform is goint to be used on image
        self.transform = transform
        
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1  #even index
        self.Y[1::2]=0  #odd index
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)     
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
               
        image=torch.load(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
if __name__ == '__main__':
	dataset = MPII('train')
	for i in range(len(dataset)):
		ii = np.random.randint(len(dataset))
		data = dataset[ii]
		print(data['image'].min())
		print(data['image'].max())        
		plt.subplot(1, 3, 1)
		#plt.imshow(data['image'].transpose(1, 2, 0)[:, :, ::-1] + 0) ## orginal line replaced by below line
		plt.imshow(data['image'].numpy().transpose(1, 2, 0)[:, :, ::-1] + 0)
		plt.subplot(1, 3, 2)
		print(data['heatmaps'].shape)
		print(data['heatmaps'].min())
		print(data['heatmaps'].max()) 
		plt.imshow(data['heatmaps'].numpy().max(0)) ## originally numpy() was  not there
		plt.subplot(1, 3, 3)
		plt.imshow(data['occlusions'].numpy().max(0)) ## originally numpy() was  not there
		plt.show()
    

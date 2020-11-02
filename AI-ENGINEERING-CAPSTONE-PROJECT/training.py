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
from model import mymodel
from dataset.dataset import Data
from util import get_default_device
torch.manual_seed(0)




parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--optimizer_type', default = 'Adam')
parser.add_argument('--mode', default = 'val')


args = parser.parse_args()
#----------------------------------------------------------------






use_cuda = True
device = torch.device("cpu")
if use_cuda:
    device = get_default_device()
    
    
    
train_dataset = Data(train=True)
validation_dataset = Data(train=False)
print("done")


batch_size = 100
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size)
validation_loader = DataLoader(dataset = validation_dataset, batch_size = batch_size)

model = mymodel("resnet18")#models.resnet18(pretrained = True)
model = model.to(device)
model

for param in model.parameters():
    param.requires_grad = False 
    
    
    
n_hidden = 512
n_out = 2

model.fc = nn.Linear(n_hidden, n_out)



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)



n_epochs=1
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()
for epoch in range(n_epochs):
    for i, (x, y) in enumerate(train_loader):
        print('-' * 30)
        print('Iteration (train phase) {}/{}'.format(i+1, int(N_train/batch_size)))        
        i_start_time = time.time()
        
        x = x.to(device)
        y = y.to(device)

        model.train() 
        
        #clear gradient 
        optimizer.zero_grad()
        
        #make a prediction 
        z = model(x)
        
        # calculate loss 
        loss = criterion(z, y)
        
        # calculate gradients of parameters 
        loss.backward()
        
        # update parameters 
        optimizer.step()
        
        loss_list.append(loss.data)
        print("Finished in {} (s)".format(time.time() - i_start_time))
 
    correct=0
    for i, (x_test, y_test) in enumerate(validation_loader):
        
        print('-' * 30)
        print("Iteration (validation phase) {} / {}".format(i + 1, int(N_test / batch_size)))
        
        i_start_time = time.time()
        
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        # set model to eval 
        model.eval()
        
        #make a prediction 
        z = model(x_test)
        
        #find max 
        _, yhat = torch.max(z.data, 1)
       
        #Calculate misclassified  samples in mini-batch 
        #hint +=(yhat==y_test).sum().item()
        correct += (yhat == y_test).sum().item()
        
        print("Finished in {} (s)".format(time.time() - i_start_time))
        
   
    accuracy=correct/N_test
    print("Epoch %d - accuracy: %.3f" % (epoch+1, accuracy))

    accuracy_list.append(accuracy)
    print("=" * 72)

    # Duration for epoch
    print("Finished epoch {} in {} (s)".format(epoch + 1, time.time() - start_time))
    
    
    

    
## testing part ##

count = 0
max_num_of_items = 4
validation_loader_batch_one = DataLoader(dataset = validation_dataset, batch_size = 1)


for i, (x_test, y_test) in enumerate(validation_loader_batch_one):

    # set model to eval
    model.eval()

    # make a prediction
    z = model(x_test)

    # find max
    _, yhat = torch.max(z.data, 1)

    # print mis-classified samples
    if yhat != y_test:
        print("Sample : {} \n Expected Label: {} \n Obtained Label:{}".format(str(i), str(y_test), str(yhat)))

        count += 1
        if count >= max_num_of_items:
            break

              




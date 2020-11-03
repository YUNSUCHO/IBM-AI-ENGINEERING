# These are the libraries will be used for this lab.
import tensorboard
import argparse
import torchvision
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
from torch.utils.tensorboard import SummaryWriter
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

parser.add_argument('--modelName', required=False, help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--batch_size', type = int, default = 100)
parser.add_argument('--optimizer_type', default = 'Adam')
parser.add_argument('--mode', default = 'val')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')


args = parser.parse_args()
#----------------------------------------------------------------


# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/experiment02')

# Function for tesor image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


use_cuda = True
device = torch.device("cpu")
if use_cuda:
    device = get_default_device()
    
    
    
train_dataset = Data(train=True)
validation_dataset = Data(train=False)
print("done")


# creat train/validation loader
batch_size = 100
train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size)
validation_loader = DataLoader(dataset = validation_dataset, batch_size = args.batch_size)


# Call the model
model = mymodel("resnet18")#models.resnet18(pretrained = True)
model = model.to(device)
model

# trasfer learning(freezed the parameters)
for param in model.parameters():
    param.requires_grad = False 
    
    
# Modify the last layer of the model    
n_hidden = 512
n_out = 2

model.fc = nn.Linear(n_hidden, n_out)



# Loss function
criterion = nn.CrossEntropyLoss()


# Optimizer
if(args.optimizer_type == 'Adam'):
    
    optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=args.lr)

elif (args.optimizer_type == 'SGD'):
    
    optimizer = torch.optim.SGD([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=args.lr)

else:
    raise NotImplementedError


    
# Training the model(traigning schedule)    
n_epochs=args.epochs
loss_list=[]
accuracy_list=[]
correct=0
N_test=len(validation_dataset)
N_train=len(train_dataset)
start_time = time.time()
#n_epochs

Loss=0
start_time = time.time()
train_count  = 0
test_count = 0
for epoch in range(n_epochs):
    for i, (x, y) in enumerate(train_loader):
        print('-' * 30)
        print('Iteration (train phase) {}/{}'.format(i+1, int(N_train/batch_size)))        
        i_start_time = time.time()
        
        # For tensorboard images #
        # create grid of images
        img_grid = torchvision.utils.make_grid(x)

        # show images
        matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('images', img_grid, i)
        
        
        x = x.to(device)
        y = y.to(device)

        model.train() 
        
        #clear gradient 
        optimizer.zero_grad()
        
        #make a prediction 
        z = model(x)
        
        # calculate loss 
        loss = criterion(z, y)
        train_count+=1
        writer.add_scalar('training loss',
                            loss,
                            train_count)
        
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
            
            # 'test-loss in tensorboard #
            testloss = criterion(z, y_test)
            test_count +=1
            writer.add_scalar('test-loss',
                                testloss,
                                test_count )
        

            #find max 
            _, yhat = torch.max(z.data, 1)

            #Calculate misclassified  samples in mini-batch 
            #hint +=(yhat==y_test).sum().item()
            correct += (yhat == y_test).sum().item()

            print("Finished in {} (s)".format(time.time() - i_start_time))
            # Correct in tensorboard #
            writer.add_scalar('correct',
                                correct,
                                test_count )


        accuracy=correct/N_test
        writer.add_scalar('accuracy',
                                  accuracy,
                                  i )
        print("Epoch %d - accuracy: %.3f" % (epoch+1, accuracy))

        accuracy_list.append(accuracy)
        print("=" * 72)

        # Duration for epoch
        print("Finished epoch {} in {} (s)".format(epoch + 1, time.time() - start_time))
        
        
    
 

    # Save model

    if i > 0 and i % 2 == 0:
                # Saving the model and the losses
                torch.save({'model': model}, \
                             os.path.join(args.modelName, 'model_{}_{}.pt'.format(epoch, i)))

    
   
# Confusion matrix # 
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()    
    
    
    

    
    
## testing part ##
from sklearn.metrics import confusion_matrix



count = 0
max_num_of_items = 4
validation_loader_batch_one = DataLoader(dataset = validation_dataset, batch_size = 1)
nb_classes = 2

# Initialize the prediction and label lists(tensors)
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
for i, (x_test, y_test) in enumerate(validation_loader_batch_one):

    # set model to eval
    model.eval()

    # make a prediction
    z = model(x_test)

    # find max
    _, yhat = torch.max(z.data, 1)
    
    predlist=torch.cat([predlist,yhat.view(-1).cpu()])
    lbllist=torch.cat([lbllist,y_test.view(-1).cpu()])

    # print mis-classified samples
    if yhat != y_test:
        print("Sample : {} \n Expected Label: {} \n Obtained Label:{}".format(str(i), str(y_test), str(yhat)))

        count += 1
        if count >= max_num_of_items:
            break
            
# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
plot_confusion_matrix(conf_mat , normalize    = False, target_names =  ['not-cracked', 'cracked'] )




# Classification report

from sklearn.metrics import classification_report
print('\n Classification Report:\n', classification_report(lbllist.numpy(), predlist.numpy()))
    

import pandas as pd
df = pd.DataFrame(classification_report(lbllist.numpy(), predlist.numpy(),
                                        output_dict=True)).T

df['support'] = df.support.apply(int)

df.style.background_gradient(cmap='viridis',
                             subset=pd.IndexSlice['0':'9', :'f1-score'])






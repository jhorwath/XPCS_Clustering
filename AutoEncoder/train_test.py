#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys

import params_Jay as params
#params.set_gpu_env()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm 
import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib

import os

from sklearn.utils import shuffle

torch.__version__


# In[2]:

plt.style.use('seaborn-white')
#matplotlib.rc('xtick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font',family='Times New Roman')
matplotlib.rcParams['font.size'] = 20
plt.viridis()


# In[3]:


def plot3(data,titles):
    if(len(titles)<3):
        titles=["Plot1", "Plot2", "Plot3"]
    fig,ax = plt.subplots(1,3, figsize=(20,12))
    im=ax[0].imshow(data[0],origin='lower')
    ax[0].set_title(titles[0],fontsize=20)
    ax[0].axis('off')
    plt.colorbar(im,ax=ax[0], fraction=0.046, pad=0.04)
    im=ax[1].imshow(data[1],origin='lower')
    ax[1].set_title(titles[1],fontsize=20)
    ax[1].axis('off')
    plt.colorbar(im,ax=ax[1], fraction=0.046, pad=0.04)
    im=ax[2].imshow(data[2],origin='lower')
    ax[2].set_title(titles[2],fontsize=20)
    ax[2].axis('off')
    plt.colorbar(im,ax=ax[2], fraction=0.046, pad=0.04)


# In[4]:


# Create path to save model
path = '/lcrc/project/AI_XPCS/XPCS_Clustering'#os.getcwd()
MODEL_SAVE_PATH = path +'/trained_model_Jay/08192022/'
if (not os.path.isdir(MODEL_SAVE_PATH)):
    os.mkdir(MODEL_SAVE_PATH)


# In[5]:


augmented_data = np.load('/lcrc/project/AI_XPCS/Clustering_Data/augmented_images.npy')
augmented_data = shuffle(augmented_data, random_state=0)
print(augmented_data.shape)
print(augmented_data.dtype)


# In[6]:


augmented_data = torch.Tensor(augmented_data)
N_DATA = augmented_data.shape[0]
N_TRAIN = int(round((1-params.TEST_FRACT-params.VALID_FRACT)*N_DATA))
N_TEST = int(round(params.TEST_FRACT*N_DATA))
N_VALID = int(round(params.VALID_FRACT*N_DATA))
print("Data, Train, test, valid split:", N_DATA, N_TRAIN, N_TEST, N_VALID)
train_data, valid_data, test_data =     torch.utils.data.random_split(augmented_data,[N_TRAIN, N_VALID, N_TEST])
print(len(train_data), len(valid_data), len(test_data))


# In[7]:


trainloader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=4)

validloader = DataLoader(valid_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=4)

testloader = DataLoader(test_data, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=4)


# In[8]:


from turtle import forward


nconv = params.nconv
drop = params.drop
pool = params.pool
#latent_size = 2#params.latent_size
latent_size = sys.argv[1]

class recon_model(nn.Module):

    def __init__(self):
        super(recon_model, self).__init__()


        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
          nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.MaxPool2d((pool,pool)),

          nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),          
          nn.ReLU(),
          nn.MaxPool2d((pool,pool)),

          nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),     
          nn.Dropout(drop),     
          nn.ReLU(),
          nn.MaxPool2d((pool,pool)),

          #nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          #nn.Dropout(drop),
          #nn.ReLU(),
          #nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),     
          #nn.Dropout(drop),     
          #nn.ReLU(),
          #nn.MaxPool2d((pool,pool)),
        )


        self.bottleneck = nn.Sequential(
          # FC layer at bottleneck -- dropout might not make sense here
          nn.Flatten(),
          nn.Linear(self.calc_fc_shape(), latent_size),
          #nn.Dropout(drop),
          nn.ReLU(),
          nn.Linear(latent_size, self.flattened_size),
          #nn.Dropout(drop),
          nn.ReLU(),
          nn.Unflatten(1,self.conv_bock_output_shape)# 0 is batch dimension
          )


        self.decoder1 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Upsample(scale_factor=pool, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Upsample(scale_factor=pool, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.Dropout(drop),
          nn.ReLU(),
          nn.Upsample(scale_factor=pool, mode='bilinear'),
            
          #nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          #nn.Dropout(drop),
          #nn.ReLU(),
          #nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          #nn.Dropout(drop),
          #nn.ReLU(),
          #nn.Upsample(scale_factor=pool, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)), #Output conv layer has 2 for mu and sigma
          nn.Sigmoid() #Amplitude model
          )
    

    def forward(self,x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            x1 = self.bottleneck(x1)
            #print(x1.shape)
            return self.decoder1(x1)


    #Helper function to calculate size of flattened array from conv layer shapes    
    def calc_fc_shape(self):
        x0 = torch.zeros([params.H,params.W]).unsqueeze(0)
        x0 = self.encoder(x0)

        self.conv_bock_output_shape = x0.shape
        #print ("Output of conv block shape is", self.conv_bock_output_shape)
        self.flattened_size = x0.flatten().shape[0]
        #print ("Flattened layer size is", self.flattened_size)
        return self.flattened_size


# In[9]:


model = recon_model()
for images in trainloader:
    print("batch size:", images.shape)
    output = model(images)
    print(output.shape)
    print(output.dtype)
    break


# In[10]:


summary(model,(1,params.H,params.W),device="cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model) #Default all devices

model = model.to(device)


# In[11]:


#Optimizer details
iterations_per_epoch = np.floor((N_TRAIN-N_VALID)/params.BATCH_SIZE)+1 #Final batch will be less than batch size
step_size = 6*iterations_per_epoch #Paper recommends 2-10 number of iterations, step_size is half cycle
print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = params.LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=params.LR / 10, max_lr=params.LR, step_size_up=step_size,
                                              cycle_momentum=False, mode='triangular2')


# In[12]:


#Function to update saved model if validation loss is minimum
def update_saved_model(model, path):
    if not os.path.isdir(path):
        os.mkdir(path)
#    for f in os.listdir(path):
#        os.remove(os.path.join(path, f))
    if (params.NGPUS>1):    
        torch.save(model.module.state_dict(),path+'best_model_{:04d}.pth'.format(latent_size)) #Have to save the underlying model else will always need 4 GPUs
    else:
        torch.save(model,path+'best_model_{:04d}.pth'.format(patent_size))


# In[13]:


scaler = torch.cuda.amp.GradScaler()


# In[14]:


def train(trainloader,metrics):
    tot_loss = 0.0
    
    for i, images in tqdm(enumerate(trainloader)):
        images = images.to(device) #Move everything to device

        preds = model(images) #Forward pass

        #Compute losses
        loss = criterion(preds,images)
        
        #Zero current grads and do backprop
        optimizer.zero_grad() 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        #optimizer.step()

        tot_loss += loss.detach().item()

        #Update the LR according to the schedule -- CyclicLR updates each batch
        scheduler.step() 
        metrics['lrs'].append(scheduler.get_last_lr())
        scaler.update()
        
        
    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([tot_loss/i]) 
    

def validate(validloader,metrics):
    tot_val_loss = 0.0

    for j, images in enumerate(validloader):
        images = images.to(device)

        preds = model(images) #Forward pass
    
        val_loss = criterion(preds,images)
           
        tot_val_loss += val_loss.detach().item()

    metrics['val_losses'].append([tot_val_loss/j])
  
  #Update saved model if val loss is lower
    if(tot_val_loss/j<metrics['best_val_loss']):
        print("Saving improved model after Val Loss improved from %.5f to %.5f" %(metrics['best_val_loss'],tot_val_loss/j))
        metrics['best_val_loss'] = tot_val_loss/j
        update_saved_model(model, MODEL_SAVE_PATH)


# In[15]:


metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}


# In[16]:


update_saved_model(model,MODEL_SAVE_PATH)


# In[ ]:


for epoch in range (params.EPOCHS):
    
  #Set model to train mode
  model.train() 
    
  #Training loop
  train(trainloader,metrics)

    
  #Switch model to eval mode no more
  #model.eval()
    
  #Validation loop
  validate(validloader,metrics)
  
  print('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
  #print('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' %(epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
  #print('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' %(epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
  print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0]))
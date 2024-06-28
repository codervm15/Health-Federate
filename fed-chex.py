#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import time
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image
from barbar import Bar
import datetime
import time
from tqdm import tqdm 
from self_supervised.tasks import fast_classification


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


# In[ ]:


import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()
print(use_gpu)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


class Config(object):
    def __init__(self):
        self.name = 'fed-chex_res18'
        self.dataset_name = 'Chexpert'
        self.save_path = './checkpoint/' + self.name
        self.model_path = self.save_path + '/models'

        self.num_threads = 8
        self.shuffle_dataset=True
        self.random_seed=24
        self.shuffle = False

        self.lr = 0.002      
        
        self.serial_batches = False
        self.phase='train'
        
        self.batch_size = 16
        self.test_batch_size = 1
        self.valid_size = 0.04
        self.test_size  = 0.5
        self.train_size = 0.1
        self.num_classes = 13
        
        self.max_epochs = 200
 

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
opt = Config()


# In[ ]:


Traindata = pd.read_csv('/workspace/DATASETS/CheXpert-v1.0-small/train.csv')

Traindata = Traindata[Traindata['Path'].str.contains("frontal")] # use only frontal images
# Traindata = Traindata[500:]
# Traindata.to_csv('/workspace/DATASETS/CheXpert-v1.0-small/train_mod.csv', index = False)
# print("Train data length:", len(Traindata))

Validdata = pd.read_csv('/workspace/DATASETS/CheXpert-v1.0-small/valid.csv')
Validdata = Validdata[Validdata['Path'].str.contains("frontal")] # use only frontal images
# Validdata.to_csv('/workspace/DATASETS/CheXpert-v1.0-small/valid_mod.csv', index = False)
# print("Valid data length:", len(Validdata))

Testdata = Traindata.head(25000) # use first 500 training data as test data (obs ratio is almost same!)
# Testdata.to_csv('/workspace/DATASETS/CheXpert-v1.0-small/test_mod.csv', index = False)
# print("Test data length:", len(Testdata))

pathFileTrain = '/workspace/DATASETS/CheXpert-v1.0-small/train_mod.csv'
pathFileValid = '/workspace/DATASETS/CheXpert-v1.0-small/valid_mod.csv'
pathFileTest = '/workspace/DATASETS/CheXpert-v1.0-small/test_mod.csv'


# In[ ]:


# csv = pd.read_csv('/workspace/DATASETS/CheXpert-v1.0-small/train.csv')
# csv = csv[csv['Path'].str.contains("frontal")]   # use only frontal images
# csv = csv.sample(frac = 1)
# print("Total data length:", len(csv))
# Traindata = csv[:170000]
# print("Train data length:", len(Traindata))
# Testdata = csv[170000:]
# print("Test data length:", len(Testdata))
# Validdata = pd.read_csv('/workspace/DATASETS/CheXpert-v1.0-small/valid_mod.csv')
# Validdata = Validdata[Validdata['Path'].str.contains("frontal")]
# print("Valid data length:", len(Validdata))

# # Traindata.to_csv('/workspace/DATASETS/CheXpert-v1.0-small/train_mod.csv', index = False)
# # Validdata.to_csv('/workspace/DATASETS/CheXpert-v1.0-small/valid_mod.csv', index = False)
# # Testdata.to_csv('/workspace/DATASETS/CheXpert-v1.0-small/test_mod.csv', index = False)

# pathFileTrain = '/workspace/DATASETS/CheXpert-v1.0-small/train_mod.csv'
# pathFileValid = '/workspace/DATASETS/CheXpert-v1.0-small/valid_mod.csv'
# pathFileTest = '/workspace/DATASETS/CheXpert-v1.0-small/test_mod.csv'


# In[ ]:


# Neural network parameters:
nnIsTrained = False     # pre-trained using ImageNet

# Training settings: batch size, maximum number of epochs
trBatchSize = 16
trMaxEpoch = 3


nnIsTrained = False     # pre-trained using ImageNet
nnClassCount = 14       # dimension of the output

imgtransResize = (320, 320)
imgtransCrop = 224
trMaxEpoch = 500

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


# In[ ]:


class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                label = line[5:]
                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


# In[ ]:


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)
# Tranform data
normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
transformList = []
transformList.append(transforms.Resize((imgtransCrop, imgtransCrop)))
transformList.append(transforms.RandomResizedCrop(imgtransCrop))
transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence = transforms.Compose(transformList)


# Load dataset
datasetTrain = CheXpertDataSet(pathFileTrain, transformSequence, policy = "ones")
print("Train data length:", len(datasetTrain))

datasetValid = CheXpertDataSet(pathFileValid, transformSequence)
print("Valid data length:", len(datasetValid))

datasetTest = CheXpertDataSet(pathFileTest, transformSequence, policy = "ones")
print("Test data length:", len(datasetTest))


# In[ ]:


# train_loader = DataLoader(datasetTrain,batch_size=opt.batch_size,shuffle=True,num_workers=8,pin_memory=True)
# val_loader = DataLoader(datasetValid,batch_size=opt.batch_size,shuffle=True,num_workers=8,pin_memory=True)
# test_loader = DataLoader(datasetTest,batch_size=opt.batch_size,shuffle=True,num_workers=8,pin_memory=True)



# FOR MULTIPLE COMMUNICATION ROUNDS
com_round = 3
fraction = 1.0
epoch = 3
batch = 48
num_clients = 5

'''
# Divide datasetTrain_ex
datasetTrain_1, datasetTrain_2, datasetTrain_3, datasetTrain_4, datasetTrain_5, dataleft = random_split(datasetTrain, 
                                                                                                        [100, 100, 100, 100, 100,
                                                                                                         len(datasetTrain) - 500])
'''
# Divide datasetTrain_real
datasetTrain_1, datasetTrain_2, datasetTrain_3, datasetTrain_4, datasetTrain_5 = random_split(datasetTrain, 
                                                                                              [34000,34000,34000,34000,34000])


# Define 5 DataLoaders
dataLoaderTrain_1 = DataLoader(dataset = datasetTrain_1, batch_size = trBatchSize,
                               shuffle = True, num_workers = 8, pin_memory = True)
dataLoaderTrain_2 = DataLoader(dataset = datasetTrain_2, batch_size = trBatchSize,
                               shuffle = True, num_workers = 8, pin_memory = True)
dataLoaderTrain_3 = DataLoader(dataset = datasetTrain_3, batch_size = trBatchSize,
                               shuffle = True, num_workers = 8, pin_memory = True)
dataLoaderTrain_4 = DataLoader(dataset = datasetTrain_4, batch_size = trBatchSize,
                               shuffle = True, num_workers = 8, pin_memory = True)
dataLoaderTrain_5 = DataLoader(dataset = datasetTrain_5, batch_size = trBatchSize,
                               shuffle = True, num_workers = 8, pin_memory = True)

# Define Valid and Test DataLoaders
dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = trBatchSize, 
                           shuffle = False, num_workers = 8, pin_memory = True)
dataLoaderTest = DataLoader(dataset = datasetTest, num_workers = 8, pin_memory = True)
dT = [datasetTrain_1, datasetTrain_2, datasetTrain_3, datasetTrain_4, datasetTrain_5]
dLT = [dataLoaderTrain_1, dataLoaderTrain_2, dataLoaderTrain_3, dataLoaderTrain_4, dataLoaderTrain_5]


# In[ ]:


class ResNet18(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard ResNet50
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained = False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(nn.Linear(num_ftrs, out_size),
                                        nn.Sigmoid()
                                        )

    def forward(self, x):
        x = self.resnet18(x)
        return x
    
model = ResNet18(nnClassCount).to(device)
checkpoint = None


# In[ ]:


def computeAUROC(dataGT, dataPRED, classCount):
    # Computes area under ROC curve 
    # dataGT: ground truth data
    # dataPRED: predicted data
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return outAUROC


# In[ ]:


def epochTrain(model, dataLoaderTrain, optimizer, epochMax, classCount, loss):
    losstrain = 0
    model.train()

    for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):
        varTarget = target.cuda(non_blocking = True)
        varInput = varInput.cuda(non_blocking = True)
        varOutput = model(varInput)
        lossvalue = loss(varOutput, varTarget)

        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()

        losstrain += lossvalue.item()

    return losstrain / len(dataLoaderTrain)


# In[ ]:


def epochVal(model, dataLoaderVal, optimizer, epochMax, ClassCount, loss):
    model.eval()
    lossVal = 0
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    print('classCount :',ClassCount)

    with torch.no_grad():
        for i, (varInput, target) in enumerate(dataLoaderVal):
            
            target = target.cuda(non_blocking = True)
            outGT = torch.cat((outGT, target),0)
            outGT = outGT.cuda(non_blocking = True)
            varInput = varInput.cuda(non_blocking = True)
            varOutput = model(varInput)
            outPRED = torch.cat((outPRED,varOutput), 0)
            lossVal += loss(varOutput, target)
        aurocIndividual = computeAUROC(outGT, outPRED, ClassCount)
        aurocMean = np.array(aurocIndividual).mean()
#         print('AUROC mean ', aurocMean)

    return lossVal / len(dataLoaderVal),aurocIndividual,aurocMean


# In[ ]:


def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint):
    optimizer = optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0) 
    loss = torch.nn.BCELoss() 

    if checkpoint != None and use_gpu: # loading checkpoint
        modelCheckpoint = torch.load(checkpoint)
        model.load_state_dict(modelCheckpoint['state_dict'])
        optimizer.load_state_dict(modelCheckpoint['optimizer'])

    # Train the network
    lossMIN = 100000
    train_start = []
    train_end = []
    for epochID in range(0, trMaxEpoch):
        train_start.append(time.time()) # training starts
        losst = epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss)
        train_end.append(time.time()) # training ends
        lossv,aurocIndividual,aurocMean = epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
        
        print("Training loss: {:.3f},".format(losst), "Valid loss: {:.3f}".format(lossv),"Valid auc: {:.3f}".format(aurocMean))
        if lossv < lossMIN:
            lossMIN = lossv
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                        'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                        os.path.join(opt.model_path,'resnet50_m-epoch' + str(epochID + 1) + '.pth.tar'))
            
            print('Epoch ' + str(epochID + 1) + ' [save] loss = ' + str(lossv.item()))
        else:
            print('Epoch ' + str(epochID + 1) + ' [----] loss = ' + str(lossv.item()))

    train_time = np.array(train_end) - np.array(train_start)
    print("Training time for each epoch: {} seconds".format(train_time.round(0)))
    params = model.state_dict()
    return params


# In[ ]:


def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):
    cudnn.benchmark = True
    use_gpu = True
    if checkpoint != None:
        modelCheckpoint = torch.load(checkpoint,map_location=device)
        model.load_state_dict(modelCheckpoint['state_dict'])
#         model = model.to(device)
    if use_gpu:
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
    else:
        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(dataLoaderTest):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0).cuda()
            bs, c, h, w = input.size()
            varInput = input.view(-1, c, h, w).cuda()

            out = model(varInput)
            outPRED = torch.cat((outPRED, out), 0)
    aurocIndividual = computeAUROC(outGT, outPRED, nnClassCount)
    aurocMean = np.array(aurocIndividual).mean()
    print('AUROC mean ', aurocMean)

    for i in range (0, len(aurocIndividual)):
        print(class_names[i], ' ', aurocIndividual[i])

    return outGT, outPRED


# In[ ]:


for i in range(com_round):
    print("[[[ Round {} Start ]]]".format(i + 1))
    params = [None] * num_clients
    sel_clients = sorted(random.sample(range(num_clients),round(num_clients*fraction))) # Step 1: select random fraction of clients
    print("The number of clients:", len(sel_clients))
    
    for j in sel_clients: # Step 2: send weights to clients
        print("<< Client {} Training Start >>".format(j + 1))
        train_valid_start = time.time()
        params[j] = train(model, dLT[j], dataLoaderTest, # Step 3: Perform local computations
                                          nnClassCount, trMaxEpoch = epoch, checkpoint = None)
        train_valid_end = time.time()
        client_time = round(train_valid_end - train_valid_start)
        print("<< Client {} Training End: {} seconds elapsed >>".format(j + 1, client_time))
        
    fidx = [idx for idx in range(len(params)) if params[idx] != None][0]
    lidx = [idx for idx in range(len(params)) if params[idx] != None][-1]
    
    for key in params[fidx]: # Step 4: return updates to server
        weights, weightn = [], []
        for k in sel_clients:
            weights.append(params[k][key]*len(dT[k]))
            weightn.append(len(dT[k]))
        params[lidx][key] = sum(weights) / sum(weightn) # weighted averaging model weights

    model = ResNet18(nnClassCount).to(device)
    model.load_state_dict(params[lidx]) # Step 5: server updates global state
    print("[[[ Round {} End ]]]".format(i + 1))
    
print("Global model trained")


# In[ ]:





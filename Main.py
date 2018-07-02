
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms

import cv2
from matplotlib import pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import time
import pandas as pd


# In[2]:


cuda = True

sad = 128
obd = 64
fd = 64
scaling_factor = 1
bs = 32


# In[3]:


import copy

model_vgg = models.vgg16(pretrained=True)

for param in model_vgg.parameters():
    param.requires_grad = False
model_vgg


# In[4]:



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = copy.deepcopy(model_vgg.features[0])
        self.conv2 = copy.deepcopy(model_vgg.features[2])
        self.conv3 = copy.deepcopy(model_vgg.features[5])
        self.conv4 = nn.Conv2d(128, 256, 5, stride=1, padding = 2)
        

        self.convs1 = copy.deepcopy(model_vgg.features[0])
        self.convs2 = copy.deepcopy(model_vgg.features[2])
        self.convs3 = copy.deepcopy(model_vgg.features[5])
        self.convs4 = nn.Conv2d(128, 256, 5, stride=1, padding = 2)
        
        
        # an affine operation: y = Wx + b
        image_size = fd/8
        
        self.fc1 = nn.Linear(256 * image_size * image_size * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x1 ,x2):

        x1 = F.relu(self.conv1(x1))
        x1 = F.max_pool2d(F.relu(self.conv2(x1)), 2)
        x1 = F.max_pool2d(F.relu(self.conv3(x1)), 2)
        x1 = F.max_pool2d(F.relu(self.conv4(x1)), 2)
        
        x2 = F.relu(self.convs1(x2))
        x2 = F.max_pool2d(F.relu(self.convs2(x2)), 2)
        x2 = F.max_pool2d(F.relu(self.convs3(x2)), 2)
        x2 = F.max_pool2d(F.relu(self.convs4(x2)), 2)
        
        
        x = torch.cat((x1, x2))
        x = x.view(-1, self.num_flat_features(x)*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

if cuda:
    net = net.cuda()
print(net)


# In[27]:


# img = cv2.imread('example.jpg',1)
img = cv2.imread('details.jpg',1)

def crop(img, lu, rb, trans=False):
    w = img.shape[0]
    h = img.shape[1]
    try:
        lu = list(map(int, lu))
        rb = list(map(int, rb))

        if lu[0] < 0:
            lu[0] = 0 

        if lu[1] < 0:
            lu[1] = 0 

        if rb[0] > img.shape[1]:
            rb[0] = img.shape[1]

        if rb[1] > img.shape[0]:
            rb[1] = img.shape[0] 

        if trans:
            img.transpose(1, 0, 2)[lu[0]:rb[0], lu[1]:rb[1]].transpose(1,0,3)
        return img[lu[1]:rb[1], lu[0]:rb[0]]
    except Exception as e:
        return np.array([])
    
def resize(img, h, w):
    return cv2.resize(img, (w, h)) 

def random_crop(sample, m=40):
    h = sample.shape[1]
    w = sample.shape[0]
    mw = m
    mh = m
    
    l = random.randint(0, w-mw)
#     r = random.randint(l+mw, w)
    r = l + mw
    
    u = random.randint(0, h-mh)
#     b = random.randint(u+mh, h)
    b = u + mh
    
    return crop(sample, (l, u), (r, b)), ((l, u), (r, b))
    
def display_boxes(img, boxL, boxT, boxO=None, re=False):
#     print(boxL)
#     print(boxT)
    
    t = cv2.rectangle(img,boxL[0],boxL[1],(255,0,0), 1)
    t = cv2.rectangle(t,boxT[0],boxT[1],(0,255,0), 1)
    if boxO is not None:
        t = cv2.rectangle(t,boxO[0],boxO[1],(0,0,255), 1)
    
    t = t if not re else resize(t, 640, 640)
    cv2.imshow('image', t)

def display(img, L, T, O=None, re=False):
    print(L, T)
    h = img.shape[0]
    w = img.shape[1]
    
    boxL = center_to_box(*L)
    boxT = center_to_box(*T)
    boxO = O if O is None else center_to_box(*O)
    display_boxes(img, boxL, boxT, boxO, re)
    
def box_to_center(lu, rb):
    w = rb[0] - lu[0]
    h = rb[1] - lu[1]
    cx = (rb[0] + lu[0]) / 2
    cy = (rb[1] + lu[1]) / 2
    return cx, cy, w, h

    
def center_to_box(cx, cy, w, h, dtype=int):
    lu = dtype((cx - w/2)), dtype((cy - h/2))
    rb = dtype((cx + w/2)), dtype((cy + h/2))
    return lu, rb

    
# img = crop(img, (20, 20), (40, 100))
# img = resize(img, 20, 40)
print(img.shape)

tmp = np.array(img[:, :, 2])
img[:, :, 2] = img[:, :, 0] 
img[:, :, 0] = tmp

# img, _ = random_crop(img)
plt.imshow(img)
# display(img, (0, 0, 0, 0), (0, 0, 0, 0))
# if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()


# In[20]:


# caseNo = 1000
# sample = img


# losses = []
# cv2.destroyAllWindows()

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.0001)

# # loss = nn.MSELoss()
# loss = nn.L1Loss()

# tim = time.time()

# for i in tqdm(range(caseNo)):
#     search_area, _ = random_crop(sample, sad)
#     search_area = resize(search_area, sad, sad)
#     obj, box = random_crop(search_area, obd)

#     labels = list(map(lambda x: x/ 1, box_to_center(*box)))
    
#     X1 = resize(search_area, fd, fd)
#     X2 = resize(obj, fd, fd)
    
#     X1 = torch.tensor(([X1.transpose(2,0,1)/255])).float()
#     X2 = torch.tensor(([X2.transpose(2,0,1)/255])).float()
    
#     X1 = torch.autograd.Variable(X1, requires_grad=True)
#     X2 = torch.autograd.Variable(X2, requires_grad=True)
    
#     Y = net(X1, X2) * scaling_factor
#     T = torch.tensor([labels], requires_grad=False)
    
#     l = loss(Y, T)

#     losses.append(l)
# #     if i % 50 == 0:
#     if time.time() - tim > 2:
#         tim = time.time()
# #         torch.save(net.state_dict(), "net")

#         display(search_area, Y.data.numpy()[0], T.data.numpy()[0])
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break
#         if i % 1000 == 0:
#             t = list(map(lambda x:float(x.data.numpy()), losses))[-3000:]
#             pd.Series(t).plot()
#             plt.show()
    
#     l.backward()
#     optimizer.step()

# cv2.destroyAllWindows()

# t = map(lambda x:float(x.data.numpy()), losses)
# pd.Series(t).plot()


# In[21]:


# t = map(lambda x:float(x.data.numpy()), losses)
# pd.Series(t).plot()


# In[16]:


# len(list(net.parameters()))

        


# Scratch Area

# In[17]:


def load(vid):
    open('..//dataset/data/01-Light/01-Light_video00001/00000001.jpg')


# In[29]:


import glob
import os
import random

nob = 0

# def play(folder):

#     ann = folder.replace('/data/', '/annotations/', 1)+".ann"
#     df = pd.DataFrame(list(map(lambda x: x.strip().split() ,open(ann).readlines())))

#     df = df[[0, 3, 2, 1, 6]]
#     df. columns = ['f', 'l', 'u', 'r', 'b']

#     df['f'] = df['f'].astype(int)
#     df['l'] = df['l'].astype(float)
#     df['u'] = df['u'].astype(float)
#     df['r'] = df['r'].astype(float)
#     df['b'] = df['b'].astype(float)
#     t = dict(map(lambda x: (int(x[1]['f']), ((int(x[1]['l']), int(x[1]['u'])), (int(x[1]['r']), int(x[1]['b'])))), df.iterrows())
#     b = ((0,0),(0,0))

#     for f in sorted(glob.glob(folder+'/*')):
#         ind = int(os.path.split(f)[-1].split('.')[0])
#         img = cv2.imread(f,1)

        
#         b = t.get(ind, b)
#         display_boxes(img, b, b)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break

#         time.sleep(0.1)


def load_annotations(folder):
    ann = folder.replace('/data/', '/annotations/', 1)+".ann"
    df = pd.DataFrame(list(map(lambda x: x.strip().split() ,open(ann).readlines())))

    df = df[[0, 3, 2, 1, 6]]
    df. columns = ['f', 'l', 'u', 'r', 'b']
    df['f'] = df['f'].astype(int)
    df['l'] = df['l'].astype(float)
    df['u'] = df['u'].astype(float)
    df['r'] = df['r'].astype(float)
    df['b'] = df['b'].astype(float)
    keys = sorted(df['f'])
    # keys[-1]
    remaining = pd.DataFrame({'f': list(set(range(1, keys[-1]))-set(keys))})
    df = pd.concat([df, remaining], axis=0)
    # remaining
    df = df.sort_values(['f'])
    df = df.set_index('f')
    df = df.interpolate()

    df['cx'] = (df['l'] + df['r'])/2
    df['cy'] = (df['u'] + df['b'])/2

    df['w'] = abs(df['r'] - df['l'])
    df['h'] = abs(df['b'] - df['u'])

    # df = df[['cx', 'cy', 'h', 'w']]

    chosen = random.sample(list(df.index[:-1]), 10)
    next_chosen = map(lambda x:x+1, chosen)
    # df[chosen]
    chosen_df = df.loc[chosen]
    chosen_df = chosen_df.reset_index()
    new_chosen_df = df.loc[next_chosen]
    new_chosen_df = new_chosen_df.reset_index()
    new_chosen_df.columns = ['n' + i for i in new_chosen_df.columns]
    df = pd.concat([chosen_df, new_chosen_df], axis=1)


    df['tcx'] = (df['ncx'] - df['cx']) / df['w']
    df['tcy'] = (df['ncy'] - df['cy']) / df['h']
    df['tw'] = df['nw'] / df['w']
    df['th'] = df['nh'] / df['h']
    
    df['f'] = df['f'].astype(int)
    df['nf'] = df['nf'].astype(int)
    
    return df.fillna(0)
    
k = 2
def load(folder, show=False):
    df = load_annotations(folder)
    data = []
    to_append = 0
    images = []
#     print(len(df.index))
    for _, row in df.iterrows():
        img = int(row['f'])
        img = os.path.join(folder, "%08d.jpg"%img)
        img = cv2.imread(img,1)#.transpose(1, 0, 2)
        oimg = img.copy()
                           
        nimg = int(row['nf'])
        nimg = os.path.join(folder, "%08d.jpg"%nimg)
        nimg = cv2.imread(nimg,1)#.transpose(1, 0, 2)
        onimg = nimg.copy()
        
        images.append((oimg, onimg))
        
        if show:
            display(onimg, (row['ncx'], row['ncy'], row['nw'], row['nh']), (row['cx'], row['cy'], row['w'], row['h']))
            time.sleep(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
#                 break
        
        
        img = crop(img, (row['cx'] - row['w']/2, row['cy'] - row['h']/2), (row['cx'] + row['w']/2, row['cy'] + row['h']/2))
#         print(img.shape, row[['l', 'r', 'b','u']])
        if not all(img.shape):
            print(img.shape)
            to_append += 1
            continue
        img = resize(img, fd, fd)
        
        
        nimg = crop(nimg, (row['cx'] - row['w']/2*k, row['cy'] - row['h']/2*k), (row['cx'] + row['w']/2*k, row['cy'] + row['h']/2*k))
        
#         if show:
#             display(nimg, (0,0,0,0), (0,0,0,0))
#             time.sleep(1)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break
        nimg = resize(nimg, fd, fd)
       
        
        data.append((img, nimg, (row['tcx'], row['tcy'], row['tw'], row['th'])))        
    
    if not data:
        print(folder)
#         data, df, images = load(folder)
    data.extend(random.sample(data*bs,to_append))
    return data, df, images

folder = "../dataset/data/09-Confusion/09-Confusion_video00016"
batch = load(folder, False)

# b = ((0,0),(0,0))
# for f in sorted(glob.glob(folder+'/*')):
#     ind = int(os.path.split(f)[-1].split('.')[0])
#     img = cv2.imread(f,1)


#     b = t.get(ind, b)
#     display_boxes(img, b, b)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break


def get_batch():
    f = glob.glob('../dataset/data/*')
    folder = random.choice(glob.glob(random.choice(f)+"/*"))
    batch, ann, images = load(folder)
    X1, X2, Y = zip(*batch)
    X1, X2, Y = np.array(X1), np.array(X2), np.array(Y)
    return X1, X2, Y, ann, images


# In[202]:


load_annotations(folder)


# In[35]:


caseNo = 1000
sample = img


losses = []
cv2.destroyAllWindows()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.0001)

# loss = nn.MSELoss()
loss = nn.L1Loss()

tim = time.time()

for i in tqdm(range(caseNo)):
    
    X1, X2, T, ann, images = get_batch()
    
    X1 = torch.tensor((X1.transpose(0,3,1,2)/255)).float()
    X2 = torch.tensor((X2.transpose(0,3,1,2)/255)).float()
    if cuda:
        X1 = X1.cuda()
        X2 = X2.cuda()
    
    X1 = torch.autograd.Variable(X1, requires_grad=True)
    X2 = torch.autograd.Variable(X2, requires_grad=True)
    
    Y = net(X1, X2) * scaling_factor
    T = torch.tensor(T, requires_grad=False).float()
    
    if cuda:
        T = T.cuda()

    l = loss(Y, T)

    losses.append(l)
    print(l,end="")
#     if i % 50 == 0:
    if time.time() - tim > 2:
        tim = time.time()
        torch.save(net.state_dict(), "net")

        y = Y
        if cuda:
            y=Y.cpu()
        y=y.data.numpy()
        ncx = ann.loc[0]['w'] * y[0][0] + ann.loc[0]['cx']
        ncy = ann.loc[0]['h'] * y[0][1] + ann.loc[0]['cy']
        nw = ann.loc[0]['w'] * y[0][2]
        nh = ann.loc[0]['h'] * y[0][3]
        
        print(images[0][1].shape)
        display(images[0][1], (ncx, ncy, nw, nh), (ann['cx'][0], ann['cy'][0], ann['w'][0], ann['h'][0]), (ann['ncx'][0], ann['ncy'][0], ann['nw'][0], ann['nh'][0]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
#         if i % 1000 == 0:
#             t = list(map(lambda x:float(x.data.numpy()), losses))[-3000:]
#             pd.Series(t).plot()
#             plt.show()
    
    l.backward()
    optimizer.step()

cv2.destroyAllWindows()

if cuda:
    t = map(lambda x:float(x.cpu().data.numpy()), losses)
else:
    t = map(lambda x:float(x.data.numpy()), losses)
pd.Series(t).plot()


# In[ ]:


for i in range(1,33):
    play('../dataset/data/02-SurfaceCover/02-SurfaceCover_video'+ '%05d'%i)
    
#     play('../dataset/data/01-Light/01-Light_video'+ '%05d'%i)
# play('../dataset/data/01-Light/01-Light_video00002')
# play('../dataset/data/01-Light/01-Light_video00003')


# 999 [ 0.02457903  0.05061942 -0.08223155  0.10660891] [0.203125 0.46875  0.9375   0.984375]
# 

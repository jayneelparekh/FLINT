import matplotlib # Importing matplotlib for it working on remote server
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
import sklearn # All scikit modules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torch
import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir + '/pytorch-cnn-visualizations/src') # For importing cnn-visualization scripts
#import guided_backprop as Vis_gb
#import gradcam as Vis_gcam
#sys.path.append(home_dir)
#from libcpab.libcpab.pytorch import cpab # Just git cloned libcpab to the home directory

import shap

def generate_activations(model, device, test_loader):
    all_activ = 0
    all_target = 0
    flag = False
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        activ = model(data)[1].cpu().data.numpy()
        target = target.cpu().data.numpy()
        if not flag:
            flag = True
            all_activ = activ
            all_target = target
        else:
            all_activ = np.concatenate((all_activ, activ), axis=0)
            all_target = np.concatenate((all_target, target), axis=0)

    return [all_activ, all_target]


def cluster_activations_performance(clustering_algo, all_activ, all_target, Nclusters=10):
    clustering_algo.fit(all_activ)
    yc = km.predict(all_activ)

    # Corresponding classes between cluster predictions and true classes
    cluster_matrix = np.zeros([Nclusters, Nclusters])
    for i in range(all_activ.shape[0]):
        cluster_matrix[yc[i], all_target[i]] += 1

    correct = 0
    for i in range(Nclusters):
        cluster_i = cluster_matrix[i]
        correct += np.max(cluster_i)
        #print (np.max(cluster_i))
    print (correct)
    return cluster_matrix


def BatchSample_compare(inp, target, model1, model2, samp_idx=0):
    # FUnction to compare explanations generated using guided backprop for two models for a given batch
    # Under construction
    inp.requires_grad = True
    GBP1 = Vis_gb.GuidedBackprop(model1)
    GBP2 = Vis_gb.GuidedBackprop(model2)
    output1 = model1(inp)
    output2 = model2(inp)
    grad_gb1 = GBP1.generate_gradients(inp[samp_idx:samp_idx+1], int(target[samp_idx]))
    grad_gb2 = GBP2.generate_gradients(inp[samp_idx:samp_idx+1], int(target[samp_idx]))
    grad_gb1 = np.moveaxis(grad_gb1, 0, -1)
    grad_gb2 = np.moveaxis(grad_gb2, 0, -1)
    return grad_gb1, grad_gb2



#-------------------------------------------------


def extract_attr_max(gdata, n_idx=5):
    indices = np.argsort(-gdata, axis=0)[:n_idx]
    return indices

def extract_attr_class_max(gdata, all_y, expl_data, expl_pred, n_idx=1, thresh=0.1):
    n_class = all_y.max() + 1
    indices = np.zeros([n_class, n_idx, gdata.shape[1]])
    true_max = np.max(gdata, axis=0)
    for i in range(n_class):
        pos_arr = np.where(all_y == i)[0]
        gdata_class = gdata[pos_arr] # Data for the ith class
        local_idx = np.argsort(-gdata_class, axis=0)[:n_idx] 
        indices[i] = pos_arr[local_idx]
    if len(expl_data > 0):
        indices2 = np.zeros([n_class, n_idx, gdata.shape[1]]) - 1
        expl_class = np.zeros([n_class, gdata.shape[1]])
        for i in range(n_class):
            pos_arr = np.where(expl_pred == i)[0]
            expl_class[i] = expl_data[pos_arr].mean(axis=0)
            select_attr = np.where(expl_class[i] > thresh)[0]
            for attr in select_attr:
                indices2[i, :, attr] = indices[i, :, attr]
        return indices2.astype(int), expl_class 
    return indices.astype(int)


def plot_g_data(gdata, dataloader):
    # Not that dataloader should have shuffle = False
    gdata = np.load('output/' + dataset + '_output/test_g_out.npy')
    cd = 0
    all_y = []
    for batch_info in dataloader:
        #data, target, locs = batch_info[0].to(device), batch_info[1].to(device), batch_info[2].to(device)
        target = list(batch_info[1].cpu().data.numpy())                                # For MNIST
        all_y += target
    all_y = np.array(all_y)


def create_noise_batch(N, low=0.0, high=1.0, d1=1, d2=28, d3=28):
    # N is the number of images in the batch, d1, d2, d3 are dimensions, low, high are the value ranges
    return ((high - low)*torch.rand([N, d1, d2, d3]) + low)

def generate_angles(n_sample=1, min_deg=-30, max_deg=30):
    distribution = torch.distributions.Uniform(min_deg, max_deg)
    return distribution.sample((n_sample,)).data.numpy()

def apply_rotation(inp, angles):
    # Rotates inp with no. of elements  = n_elem according to rotation angles specified in the numpy array angles
    norm_mean, norm_std = np.array([0.0]), np.array([1.0])
    norm = transforms.Normalize(tuple(-1*norm_mean/norm_std), tuple(1/norm_std))
    norm2 = transforms.Normalize(tuple(norm_mean), tuple(norm_std))
    return torch.stack([ norm2(transforms.ToTensor()(transforms.ToPILImage()(norm(inp.detach().cpu()[i])).rotate(angles[i]))) for i in range(angles.shape[0])])

def transform_batch(inp, transform='diffeo'):
    # Applies the specified transform with random parameters on the batch of objects and returns the transformed batch
    batch_size = int(inp.shape[0])
    if transform == 'diffeo':
        theta = 0.5*T.sample_transformation(batch_size)
        inp_t = T.transform_data(inp, theta, outsize=tuple(inp.shape[2:]) )
    elif transform == 'rotate':
        theta = generate_angles(batch_size)
        inp_t = apply_rotation(inp, theta)
    return inp_t
#T = cpab(tess_size=[3,3], device='cuda')



#x = generate_activations(model, device, test_loader)
#all_activ = np.load('activations_fc1_fmnist.npy')
#all_target = np.load('testtarget_fmnist.npy')



import matplotlib # Importing matplotlib for it working on remote server
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import matplotlib.colors as color

from sklearn.cluster import KMeans, SpectralCoclustering
import os, sys
home_dir = os.getcwd()

#import random
#random.seed(121)
import torch # All the torch modules
#torch.cuda.set_device(1)
#torch.manual_seed(121)
#torch.cuda.manual_seed(121)
#torch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import networks
import guided_backpropagation as gb
import itertools
import numpy as np
import dataloader as dl
import utils
torch.manual_seed(21)
torch.cuda.manual_seed(21)
np.random.seed(21)
print ('Imported all modules')

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
train_shuffle = True
test_shuffle = True
train_epoch_info = []
test_epoch_info = []

dataset = 'mnist' # Options: qdraw, mnist, fmnist

if dataset == 'qdraw':
    n_classes = 10
    train_data = dl.QuickDraw(ncat=n_classes, mode='train', root_dir=home_dir + '/datasets/data_quickdraw/')
    test_data = dl.QuickDraw(ncat=n_classes, mode='test', root_dir=home_dir + '/datasets/data_quickdraw/')
    train_loader = torch.utils.data.DataLoader( dl.QuickDraw(ncat=n_classes, mode='train', root_dir=home_dir + '/datasets/data_quickdraw/'), batch_size=64, shuffle=train_shuffle, num_workers=32 )
    test_loader = torch.utils.data.DataLoader( dl.QuickDraw(ncat=n_classes, mode='test', root_dir=home_dir + '/datasets/data_quickdraw/'), batch_size=100, shuffle=test_shuffle, num_workers=16 )
elif dataset == 'mnist':
    train_data = datasets.MNIST(home_dir + '/datasets', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor() ]))
    test_data = datasets.MNIST(home_dir + '/datasets', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor() ]))
    train_loader = torch.utils.data.DataLoader( datasets.MNIST(home_dir + '/datasets', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor() ])), batch_size=16, shuffle=train_shuffle, num_workers=12)
    test_loader = torch.utils.data.DataLoader( datasets.MNIST(home_dir + '/datasets', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor() ])), batch_size=1024, shuffle=test_shuffle, num_workers=20)
elif dataset == 'fmnist':
    train_data = datasets.FashionMNIST(home_dir + '/datasets', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor() ]))
    test_data = datasets.FashionMNIST(home_dir + '/datasets', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor() ]))  
    train_loader = torch.utils.data.DataLoader( datasets.FashionMNIST(home_dir + '/datasets', train=True, download=True, transform=transforms.Compose([ transforms.ToTensor() ])), batch_size=16, shuffle=train_shuffle, num_workers=12)
    test_loader = torch.utils.data.DataLoader( datasets.FashionMNIST(home_dir + '/datasets', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor() ])), batch_size=64, shuffle=test_shuffle, num_workers=20)

print ('Dataloader ready')

criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
criterion3 = nn.BCEWithLogitsLoss()

if dataset == 'mnist' or dataset == 'fmnist':
    latent_size = 25
    f = networks.Net2_MNIST_old().to(device)
    g = networks.attr_MNIST(out_size=latent_size).to(device)
    d = networks.decode_MNIST(in_size=latent_size).to(device)
    h = networks.explainer(in_size=latent_size, n_classes=10).to(device)
    optimizer = optim.Adam(itertools.chain(f.parameters(), g.parameters(), d.parameters()), lr=0.0001)
    opt_expl = optim.Adam(itertools.chain(g.parameters(), h.parameters()), lr=0.0001)

elif dataset == 'qdraw':
    latent_size = 24
    f = networks.MyResNet(n_classes=n_classes, version='34').to(device)
    g = networks.attr_RN18_multi(out_size=latent_size).to(device)
    d = networks.decode2_MNIST(in_size=latent_size).to(device)
    h = networks.explainer(in_size=latent_size, n_classes=n_classes).to(device)
    optimizer = optim.Adam(itertools.chain(f.parameters(), g.parameters(), d.parameters(), h.parameters()), lr=0.0001)

def unnorm(img, norm_mean, norm_var):
    # img assumed as numpy array, other two as array/list like
    ndim = img.shape[0]
    new_im = 0 + img
    for i in range(ndim):
        new_im[i] = img[i] * norm_var[i] + norm_mean[i]
    return np.moveaxis(new_im, 0, -1)


def analyze(f, g, h, d, device, test_loader, location):
    f.eval(), g.eval(), h.eval(), d.eval()
    f, g, h, d = f.to(device), g.to(device), h.to(device), d.to(device)
    batch_size = int(sample_data.shape[0])
    conf_matx_fy = np.zeros([10, 10]) # n_classes x n_classes
    conf_matx_hf = np.zeros([10, 10])
    conf_matx_hy = np.zeros([10, 10])
    for batch_info in test_loader:
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        output, inter = f(data)
        embed = g(inter)
        rec_data, expl = d(embed), h(embed)
        pred_f = output.argmax(dim=1).cpu().data.numpy()
        pred_h = expl.argmax(dim=1).cpu().data.numpy()
        y = target.cpu().data.numpy()
        for j in range(y.shape[0]):
            conf_matx_fy[pred_f[j], y[j]] += 1
            conf_matx_hf[pred_h[j], pred_f[j]] += 1
            conf_matx_hy[pred_h[j], y[j]] += 1

    return conf_matx_fy, conf_matx_hf, conf_matx_hy


def collect_g_data(f, g, h, device, data, subset=False):
    f.eval(), g.eval(), h.eval()
    f, g, h = f.to(device), g.to(device), h.to(device)
    weights = h.fc1.weight.cpu().data.numpy()
    batch_size = int(sample_data.shape[0])
    g_data = []
    all_y = []
    num_batch = 0
    subset_data = []
    expl_data = [] # Only append data in this if subset is true, else it'll possibly increase the time by a lot
    expl_pred = []
    if not subset:
        dataloader = torch.utils.data.DataLoader(data, batch_size=16*4, shuffle=False, num_workers=64)
    else:
        dataloader = torch.utils.data.DataLoader(data, batch_size=20, shuffle=True, num_workers=20)
    for batch_info in dataloader:
        num_batch += 1
        #data, target, locs = batch_info[0].to(device), batch_info[1].to(device), batch_info[2].to(device)
        data, target = batch_info[0].to(device), batch_info[1].to(device)                                 # For MNIST
        output, inter = f(data)
        embed = g(inter)
        expl = np.zeros(embed.shape)
        pred = h(embed).argmax(dim=1).cpu().data.numpy()
        g_data.append(embed.cpu().data.numpy())
        all_y += list(target.cpu().data.numpy())
        expl_pred += list(h(embed).argmax(dim=1).cpu().data.numpy())
        if subset:
            subset_data.append(data)
            for i in range(pred.shape[0]):
                expl[i] = embed[i].cpu().data.numpy() * weights[pred[i]]
                expl[i] = expl[i]/expl[i].max()
            expl_data.append(expl)
            if num_batch > 50:
                subset_data = torch.cat(subset_data).unsqueeze(dim=1) #unsqueeze is done to make code in save image functions compatible with shape of subset_data
                break
    g_data = np.concatenate(g_data)
    if subset:
        expl_data = np.concatenate(expl_data)
    return g_data, np.array(all_y), subset_data, expl_data, np.array(expl_pred)

def loss_cce(prediction, target):
    # Assume shape of batch_size x n_classes with unnormalized class scores for prediction and target
    # Compute softmax to get class probabilities and then take compute cross entropy loss
    p = nn.Softmax(dim=1)(prediction)
    t = nn.Softmax(dim=1)(target)
    loss = (p.log() * -t).sum(dim=1).mean()
    return loss



def save_expl_images_class(indices, data, gdata, f, f_copy, g, device, dataset, model_name='', d=None):
    # This function assumes specific shape of indices
    if dataset == 'qdraw':
        f_gb = gb.GuidedBackprop(f)
    else:
        f_gb = f
    f_copy = f_copy.eval()
    for i in range(indices.shape[2]): # Fixing the attribute (coordinate of attribute vector)
        for j in range(indices.shape[0]): # Fixing the class
            for k in range(indices.shape[1]):
                if indices[j, k, i] == -1:
                    continue
                img = data[indices[j, k, i]][0].cpu().data.numpy()[0]
                if dataset == 'qdraw':
                    init_img = 0.1*data[indices[j, k, i]]
                else:
                    init_img = 0.2*data[indices[j, k, i]] 
                cur_img = optimize_inp(f_copy, g, i, device, list(init_img.shape), init=init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
                grad = grad_inp_embed(f_gb, g, device, data[indices[j, k, i]][0], i, dataset=dataset)
                if gdata[indices[j, k, i], i] < gdata[:, i].max()/4.0:
                    continue
                fig = plt.figure()
                fig.add_subplot(1, 5, 1)
                plt.imshow(img)
                plt.axis('off')
                fig.add_subplot(1, 5, 2)
                plt.imshow(grad)
                plt.axis('off')
                fig.add_subplot(1, 5, 3)
                attr = g(f(data[indices[j, k, i]][0].unsqueeze(0))[1])
                plt.imshow(np.moveaxis(d(attr)[0].cpu().data.numpy(), 0, -1)[:, :, 0])
                plt.axis('off')
                attr[:, i] = 0
                fig.add_subplot(1, 5, 4)
                plt.imshow(np.moveaxis(d(attr)[0].cpu().data.numpy(), 0, -1)[:, :, 0])
                plt.axis('off')
                fig.add_subplot(1, 5, 5)
                plt.imshow(cur_img[0, 0])
                plt.axis('off')
                fig.subplots_adjust(wspace=0.04) 
                plt.savefig('output/' + dataset + '_output/explanation_images_' + model_name  + '/attr' + str(i) + '_class' + str(j) + '_' + str(k), bbox_inches='tight', pad_inches = 0.03)
                plt.close()
                
    return


def analyze_img(f, f_copy, g, h, d, idx, n_attr=3, save=False, location=None):
    #f_gb = gb.GuidedBackprop(f)
    f_gb = f
    f_copy, g, d = f_copy.eval(), g.eval(), d.eval()
    if dataset == 'qdraw':
        if idx > 1000:
            init_img = test_data[idx][0].unsqueeze(0)
        else:
            init_img = sample_data[idx].unsqueeze(0)
    elif dataset == 'mnist' or datset == 'fmnist':
        init_img = sample_data[idx].unsqueeze(0)

    init_shape = init_img.shape
    img = init_img[0, 0].cpu().data.numpy()

    # Complete the computations
    output, inter = f_copy(init_img)
    embed1 = g(inter)
    weights = h.fc1.weight.cpu().data.numpy()
    pred = h(embed1).argmax(dim=1).cpu().data.numpy()
    expl1 = embed1[0].cpu().data.numpy() * weights[pred[0]]
    expl1 = expl1 / np.abs(expl1).max()
    attr_idx1 = expl1.argsort()[-1]
    attr_idx2 = expl1.argsort()[-2]
    attr_idx3 = expl1.argsort()[-3]
    cur_img1 = optimize_inp(f_copy, g, attr_idx1, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
    cur_img2 = optimize_inp(f_copy, g, attr_idx2, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
    cur_img3 = optimize_inp(f_copy, g, attr_idx3, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
    print (attr_idx1, attr_idx2, attr_idx3)
    #grad = grad_inp_embed(f_gb, g, device, init_img[0], attr_idx)

    # Make the plot
    fig = plt.figure(figsize=(16, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img)
    #plt.title('Input sample', fontsize=28)
    plt.axis('off')

    fig.add_subplot(1, 2, 2)
    x_pos = np.array(range(expl1.shape[0])).astype(str)
    x_pos = np.array(['$\phi_{'+i+'}$' for i in x_pos])
    #ax[1].plot(list(range(n_attr)), -1*np.sort(-1*expl1)[:n_attr], color='green')
    plt.bar(list(range(n_attr)), -1*np.sort(-1*expl1)[:n_attr], color='green')
    #plt.xticks(list(range(n_attr)), x_pos[(-1*expl1).argsort()[:n_attr]], fontsize=32)
    plt.xticks(list(range(n_attr)), x_pos[np.array([attr_idx1, attr_idx2, attr_idx3])], fontsize=32)
    plt.yticks([0, 0.5, 1], [0.0, 0.5, 1.0], fontsize=23)
    plt.ylabel('Relevance to prediction', fontsize=24)
    #plt.subplots_adjust(wspace=0.05)
    if not save:
        plt.show()
    else:
        plt.savefig(location + '/s' + str(idx) + '_rel_')
        plt.close()
 
    #fig.add_subplot(1, 6, 3)
    #ax[1].imshow(grad)

    #fig.add_subplot(1, 6, 4)
    #attr = g(f(init_img)[1])
    #attr[:, i] = 0
    #ax[2].imshow(np.moveaxis(d(attr)[0].cpu().data.numpy(), 0, -1)[:, :, 0])

    #attr[:, attr_idx] = 0
    #fig.add_subplot(1, 6, 5)
    #ax[3].imshow(np.moveaxis(d(attr)[0].cpu().data.numpy(), 0, -1)[:, :, 0])

    fig = plt.figure(figsize=(16, 7))
    fig.add_subplot(1, 3, 1)
    plt.imshow(cur_img1[0, 0])
    plt.axis('off')
    plt.title('$\phi_{'+str(attr_idx1)+'}$', fontsize=32)
    fig.add_subplot(1, 3, 2)
    plt.imshow(cur_img2[0, 0])
    plt.axis('off')
    plt.title('$\phi_{'+str(attr_idx2)+'}$', fontsize=32)
    fig.add_subplot(1, 3, 3)
    plt.imshow(cur_img3[0, 0])
    plt.axis('off')
    plt.title('$\phi_{'+str(attr_idx3)+'}$', fontsize=32)
    plt.subplots_adjust(wspace=0.05)
    if not save:
        plt.show()
    else:
        plt.savefig(location + '/s' + str(idx) + '_att3_')
        plt.close()
 
    return cur_img1, cur_img2

def analyze_img_attr(f, f_copy, g, d, init_img, attr_idx):
    #f_gb = gb.GuidedBackprop(f)
    f_gb = f
    f_copy, g, d = f_copy.eval(), g.eval(), d.eval()
    init_shape = init_img.shape
    img = init_img[0, 0].cpu().data.numpy()
    output, inter = f(init_img)
    embed = g(inter)

    # Complete the computations
    cur_img = optimize_inp(f_copy, g, attr_idx, device, list(init_img.shape), init=0.2*init_img, lmbd_bound=10.0, lmbd_tv=6.0, C=2.0, lmbd_l1=0.0)
    grad = grad_inp_embed(f_gb, g, device, init_img[0], attr_idx)

    # Make the plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.08)
    #fig.add_subplot(1, 6, 1)
    ax[0].imshow(img)
 
    #fig.add_subplot(1, 6, 3)
    #ax[1].imshow(grad)

    #fig.add_subplot(1, 6, 4)
    attr = g(f(init_img)[1])
    #attr[:, i] = 0
    #ax[2].imshow(np.moveaxis(d(attr)[0].cpu().data.numpy(), 0, -1)[:, :, 0])

    attr[:, attr_idx] = 0
    #fig.add_subplot(1, 6, 5)
    #ax[3].imshow(np.moveaxis(d(attr)[0].cpu().data.numpy(), 0, -1)[:, :, 0])

    #fig.add_subplot(1, 6, 6)
    #ax[4].imshow(cur_img[0, 0])
    ax[1].imshow(cur_img[0, 0])
    ax[0].axis('off')
    ax[1].axis('off')
    #ax[2].axis('off')
    #ax[3].axis('off')
    #ax[4].axis('off')
    plt.show()
    return

def plot_rel(rel, class_idx, dataset, n_attr=5):
    if dataset == 'mnist':
        class_name = [str(i) for i in range(10)]
    elif dataset == 'qdraw':
        class_name = ['Ant', 'Apple', 'Banana', 'Carrot', 'Cat', 'Cow', 'Dog', 'Frog', 'Grapes', 'Lion']
    x_pos = np.array(range(rel.shape[1])).astype(str)
    x_pos = np.array(['$\phi_{'+i+'}$' for i in x_pos])
    x_pos_int = np.array([i for i in np.array(range(rel.shape[1]))])

    he = -1*np.sort(-1*rel[class_idx])[:n_attr]
    atts = x_pos_int[(-1*rel[class_idx]).argsort()[:n_attr]]

    plt.figure(figsize=(3, 3))
    plt.bar(list(range(n_attr)), -1*np.sort(-1*rel[class_idx])[:n_attr], color='green')
    #plt.bar(list(range(n_attr)), he[atts.argsort()], color='green')
    plt.ylim(bottom=0.0, top=1.0)
    plt.xticks(list(range(n_attr)), x_pos[(-1*rel[class_idx]).argsort()[:n_attr]], fontsize=50)
    #plt.xticks(list(range(n_attr)), x_pos[np.sort(atts)], fontsize=50)
    plt.yticks([0, 0.5, 1], [0.0, 0.5, 1.0], fontsize=30) 
    #plt.xlabel('Attribute number', fontsize=14)
    plt.ylabel('Relevance to prediction', fontsize=34)
    plt.title("Top attributes for `" + class_name[class_idx] + "'", fontsize=34)  
    plt.show()


def generate_model_explanations(f, g, h, d, data, device, dataset, model_name='', subset=False):
    if not subset:
        print ('Collecting attribute vectors on the given data')
    else:
        print ('Collecting attribute vectors on random subset of the given data')
    gdata, all_y, subset_data, expl_data, expl_pred = collect_g_data(f, g, h, device, data, subset=subset)
    indices2, rel = utils.extract_attr_class_max(gdata, all_y, expl_data, expl_pred, 3, thresh=0.1)
    try:
        os.mkdir('output/' + dataset + '_output/explanation_images_' + str(model_name))	
        #os.mkdir('output/' + dataset + '_output/explanation_images_' + str(model_name) + '/inp_optimize')
    except:
        print ('Writing images in an old folder. May overwrite some files')
    print ('Saving images')
    if not subset:
        subset_data = data
    if dataset == 'qdraw':
        f_copy = networks.MyResNet(n_classes=n_classes, version='34').to(device)
    else:
        f_copy = networks.Net2_MNIST_old().to(device)
    f_copy.load_state_dict(checkpoint1['f_state_dict'])
    
    save_expl_images_class(indices2, subset_data, gdata, f, f_copy, g, device, dataset, str(model_name), d)
    return


def optimize_inp(f, g, embed_idx, device, inp_shape=[1, 1, 28, 28], init=None, max_val=1.0, min_val=0.0, lmbd_tv=1.0, lmbd_bound=1.0, C=1.0, lmbd_l1=0):
    # initialize input with input shape and make requires_grad True
    f, g = f.eval().to(device), g.eval().to(device) 
    inp = torch.empty(inp_shape).to(device)
    if init is None:
        #4.0 * (nn.init.uniform_(inp) - 0.5) # Initialization line
        nn.init.uniform_(inp)
    else:
        inp = 1.0 * init
    inp.requires_grad = True
    new_lr = 0.05
    for epoch in range(6):
        #optimizer = optim.SGD([inp], lr=new_lr, momentum=0.9)
        optimizer = optim.Adam([inp], lr=new_lr)
        new_lr = new_lr/2
        for i in range(50):
            optimizer.zero_grad()
            output, inter = f(inp)
            embed = g(inter)

            loss_l1 = (inp.abs()).mean()
            loss_bound = (( (inp > max_val).float() + (inp < min_val).float() )*(inp.abs())).mean()
            loss_tv = (inp[:, :, 0:inp_shape[2]-1, :] - inp[:, :, 1:inp_shape[2], :]).abs().mean() + (inp[:, :, :, 0:inp_shape[3]-1] - inp[:, :, :, 1:inp_shape[3]]).abs().mean()
            loss = C*embed[:, embed_idx].sum() - lmbd_l1 * loss_l1 - lmbd_bound * loss_bound - lmbd_tv * loss_tv

            loss.backward()
            inp.grad = -1 * inp.grad
            optimizer.step()
            if (i % 51 == 0 and i == 3):
                print (epoch, loss.item(), embed[:, embed_idx].sum(), loss_l1.item(), loss_bound.item(), loss_tv.item())
    return inp.cpu().data.numpy()      

    
    
'''
def test_noise_acc(f, g, h, d, device, test_loader):
    f.eval(), g.eval(), h.eval(), d.eval()
    noise_level = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 1.0]
    record = []
    for level in noise_level:
        correct = 0
        overlap = 0
        for batch_info in test_loader:
            #data, target, locs = batch_info[0].to(device), batch_info[1].to(device), batch_info[2].to(device)
            data, target = batch_info[0].to(device), batch_info[1].to(device)                                 # For MNIST
            data = data + level * utils.create_noise_batch(16, low=-1.0)
            #data = utils.transform_batch(data, 'rotate') + level * utils.create_noise_batch(16, low=-1.0)
            data = data.clamp(0.0, 1.0)
            output, inter = f(data)
            expl = h(g(inter))
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            overlap += torch.sum(o4)
            correct += pred.eq(target.view_as(pred)).sum().item()
        correct = correct * 100.0 / len(test_loader.dataset)
        overlap = overlap.item() * 100.0 / len(test_loader.dataset)
        print (level, correct, overlap)
        record.append([level, correct, overlap])
    return np.array(record)

def metric_relevancy(f, g, h, inp, device):
    # Find the order of relevancy of features and calculate their correlation with prediction probability
    f, g, h = f.eval(), g.eval(), h.eval()
    inp.requires_grad = True
    output, inter = f(inp)
    embed = g(inter)
    expl = h(embed)
    pred = output.argmax(dim=1)
    thresh = 0.3
    weights = h.fc1.weight
    obs = []
    n_pixels = []
    for samp_num in range(int(embed.shape[0])):
        pred_class = int(pred[samp_num])
        for idx in range(int(embed.shape[1])):
            #print (samp_num, idx)
            output, inter = f(inp)
            embed = g(inter)
            grad = torch.autograd.grad(embed[samp_num, idx], inp, create_graph=True, retain_graph=True)[0][samp_num]
            # Zero out values corresponding to a threshold
            inp_new = 1.0*inp[samp_num]
            inp_new[grad >= thresh * grad.abs().max()] = 0
            n_pixels.append(torch.sum(grad >= thresh * grad.abs().max()).item())
            inp_new = inp_new.unsqueeze(0)
            output_new, inter_new = f(inp_new)
            embed_new = g(inter_new)
            embed_new2 = 1.0*embed
            embed_new2[samp_num, idx] = 0.0
            expl_new = h(embed_new)
            expl_new2 = h(embed_new2)
            # Compare output_new[0] and output[samp_num]
            old_prob = output[samp_num].softmax(dim=0)
            new_prob = output_new[0].softmax(dim=0)
            old_expl_prob = expl[samp_num].softmax(dim=0)
            new_expl_prob = expl_new[0].softmax(dim=0)
            new_expl_prob2 = expl_new2[samp_num].softmax(dim=0)
            prob_drop = (old_prob[pred_class] - new_prob[pred_class]).cpu().item()
            attr_weight = weights[pred_class, idx].cpu().item()
            attr_val = embed[samp_num, idx].cpu().item()
            feat_change = ((embed[samp_num, idx] - embed_new[0, idx])*attr_weight).cpu().item()
            expl_prob_drop = (old_expl_prob[pred_class] - new_expl_prob[pred_class]).cpu().item()
            expl_prob_drop2 = (old_expl_prob[pred_class] - new_expl_prob2[pred_class]).cpu().item()
            obs.append([feat_change, expl_prob_drop, prob_drop, expl_prob_drop2, (embed[samp_num, idx]*attr_weight).cpu().item()])
            #print (feat_change, prob_drop)
    print (np.array(n_pixels).mean())
    return np.array(obs)
       

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test_adv(f, g, h, device, data, epsilon):
    test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=4)
    f, g, h = f.eval(), g.eval(), h.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []
    # Loop over all examples in test set
    for data, target in test_loader: 
        data, target = data.to(device), target.to(device) # Send the data and label to the device
        data.requires_grad = True  # Set requires_grad attribute of tensor. Important for Attack

        output, inter = f(data) # Forward pass the data through the model
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)  # Calculate the loss
        f.zero_grad()         # Zero all existing gradients
        loss.backward()           # Calculate gradients of model in backward pass
        data_grad = data.grad.data  # Collect datagrad
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output, inter = f(perturbed_data) # Re-classify the perturbed image

        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc#, adv_examples

'''


def collect_noise(f, g, device):
    f, g = f.eval(), g.eval()
    collection = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    stat = 25 * [0]
    for i in range(100):
        inp = utils.create_noise_batch(N=20)
        output, inter = f(inp)
        embed = g(inter)
        active_attr = embed.argmax(dim=1).cpu().data.numpy()
        for idx in range(active_attr.shape[0]):
            collection[active_attr[idx]].append(inp[idx][0])
            stat[active_attr[idx]] += 1
        #print ("Min:", min(stat))
        #print ("Max:", max(stat))
    return collection, stat

def grad_inp_embed(f_gb, g, device, inp, embed_idx, dataset='mnist'):
    # Assume inp of shape 1 x 28 x 28
    g = g.eval()
    inp = inp.unsqueeze(0)
    g, inp = g.to(device), inp.to(device)
    inp.requires_grad = True
    if dataset == 'qdraw':
        output, inter = f_gb.model(inp)
    else:
        output, inter = f_gb(inp)
    if dataset == 'qdraw':
        f_gb.model.zero_grad()
    else:
        f_gb.zero_grad()
    embed = g(inter)
    if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'qdraw':
        grad = torch.autograd.grad(embed[0, embed_idx], inp)[0][0, 0].cpu().data.numpy()
    elif dataset == 'cifar10':
        grad = torch.autograd.grad(embed[0, embed_idx], inp)[0][0].abs().sum(dim=0).cpu().data.numpy() # Add the code to shift axes, then remove this comment
    #print (grad.shape)
    return grad


def test(f, g, h, d, device, test_loader, lmbd_rec=0, lmbd_expl=0, lmbd_cd=0):
    f,g,h,d = f.eval(), g.eval(), h.eval(), d.eval()
    test_loss_acc = 0
    test_loss_rec = 0
    test_loss_exp = 0
    test_loss_cd = 0
    exp_overlap = 0
    vis_acc = 0
    vis_acc_1 = 0 # Visibility prediction accuracy for visible parts
    vis_acc_0 = 0 # Visibility prediction accuracy for hidden parts
    num_part_visible = 0 # Number of visible parts in all the test images
    correct = 0
    #sample_data, sample_target, sample_locs = next(iter(test_loader))
    batch_size = int(sample_data.shape[0])
    for batch_info in test_loader:
        #data, target, locs = batch_info[0].to(device), batch_info[1].to(device), batch_info[2].to(device)
        data, target = batch_info[0].to(device), batch_info[1].to(device)                                 # For MNIST
        output, inter = f(data)
        embed = g(inter)
        rec_data, expl = d(embed), h(embed)
        loss_rec = lmbd_rec * criterion2(rec_data, data.detach())
        #loss_expl = lmbd_expl * sum( [ (expl[i, target[i]] - output.detach()[i, target[i]])**2 for i in range(batch_size) ] ) / batch_size
        loss_expl = lmbd_expl * loss_cce(expl, output.detach())#lmbd_expl * criterion(expl, output.detach().argmax(dim=1))
        #loss_loc = criterion3(embed[:, :15], locs)
        #output = model(data)
        loss_ent = loss_cce(embed.abs(), embed.abs())
        loss_tran = loss_ent - 2.0*loss_cce(embed.abs().mean(dim=0).unsqueeze(0), embed.abs().mean(dim=0).unsqueeze(0))
        #loss_tran = (embed_prob * (1 - embed_prob)).sum(dim=1).mean()
        loss_spa = nn.L1Loss()(embed, torch.zeros(embed.shape).to(device))
        loss_cd = lmbd_cd * (loss_tran + 3*loss_spa)

        test_loss_acc += criterion(output, target).item()  # sum up batch loss
        test_loss_rec += loss_rec.item()
        test_loss_exp += loss_expl.item()
        test_loss_cd += loss_cd.item()
        exp_overlap += float(torch.sum(expl.argmax(dim=1) == output.argmax(dim=1)))
        #vis_acc += float( torch.sum((g(inter)[:, :15] > 0) == (locs == 1)) )
        #vis_acc_1 += float( torch.sum( ((g(inter)[:, :15] > 0) == (locs == 1)) * (locs == 1)) )
        #vis_acc_0 += float( torch.sum( ((g(inter)[:, :15] > 0) == (locs == 1)) * (locs == 0)) )
        #num_part_visible += float(torch.sum(locs == 1))
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss_acc /= (len(test_loader.dataset) / batch_size)
    test_loss_rec /= (len(test_loader.dataset) / batch_size)
    test_loss_exp /= (len(test_loader.dataset) / batch_size)
    test_loss_cd /= (len(test_loader.dataset) / batch_size)
    exp_overlap = exp_overlap * 100 / len(test_loader.dataset)
    #vis_acc = vis_acc * 100 / (15 * len(test_loader.dataset))
    #vis_acc_1 = vis_acc_1 / num_part_visible
    #vis_acc_0 = vis_acc_0 / (15 * len(test_loader.dataset) - num_part_visible)
    if dataset == 'cub':
        print('\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)   Rec: {:.5f}   Exp: {:.5f})   ExpAgree: {:.5f}   VisAcc: {:.5f}   Vis1Acc: {:.5f}   Vis0Acc: {:.5f}\n'.format(
        test_loss_acc, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), test_loss_rec, test_loss_exp, exp_overlap, vis_acc, vis_acc_1, vis_acc_0))
        return [100. * correct / len(test_loader.dataset), test_loss_acc, test_loss_rec, test_loss_exp, exp_overlap, vis_acc, vis_acc_1, vis_acc_0]
    elif dataset == 'mnist' or dataset == 'fmnist' or dataset == 'qdraw':
        print('\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)   Rec: {:.5f}   Exp: {:.5f})   ExpAgree: {:.5f}\n'.format(
        test_loss_acc, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), test_loss_rec, test_loss_exp, exp_overlap))
        return [100. * correct / len(test_loader.dataset), test_loss_acc, test_loss_rec, test_loss_exp, test_loss_cd, exp_overlap]


def train(f, g, h, d, device, train_loader, optimizer, epoch, lmbd_acc=1.0, lmbd_rec=0, lmbd_expl=0, lmbd_spa=0, lmbd_ent=0):
    # f denotes the main classifier
    # g maps intermediate layer of f to attribute space: The Variable g in this code is same as attribute function Phi
    train_loss_acc = 0.0
    train_loss_rec = 0.0
    train_loss_exp = 0.0
    train_loss_ent = 0.0
    train_loss_spa = 0.0
    f.train(), g.train(), h.train(), d.train()
    print (lmbd_acc, lmbd_rec, lmbd_expl, lmbd_spa, lmbd_ent)
    for batch_idx, batch_info in enumerate(train_loader):
        data, target = batch_info[0].to(device), batch_info[1].to(device)
        batch_size = int(target.shape[0])
        optimizer.zero_grad()
        output, inter = f(data)       
        embed = g(inter)
        rec_data, expl = d(embed), h(embed)

        loss_acc = criterion(output, target)
        loss_rec = criterion2(rec_data, data)
        loss_expl = loss_cce(expl, output.detach())
        loss_ent1 = loss_cce(embed.abs(), embed.abs())
        loss_ent = loss_ent1 - 1.0*loss_cce(embed.abs().mean(dim=0).unsqueeze(0), embed.abs().mean(dim=0).unsqueeze(0))
        loss_spa = nn.L1Loss()(embed, torch.zeros(embed.shape).to(device))
        loss = lmbd_acc*loss_acc + lmbd_rec*loss_rec + lmbd_expl*loss_expl  + lmbd_spa*loss_spa + lmbd_ent*loss_ent
        loss.backward()
        optimizer.step()

        train_loss_acc += loss_acc.item()
        train_loss_rec += loss_rec.item()
        train_loss_spa += loss_spa.item()
        train_loss_ent += loss_ent.item()
        train_loss_exp += loss_expl.item()

    train_loss_acc = train_loss_acc / (len(train_loader.dataset) / batch_size)
    train_loss_rec = train_loss_rec / (len(train_loader.dataset) / batch_size)
    train_loss_exp = train_loss_exp / (len(train_loader.dataset) / batch_size)
    train_loss_spa = train_loss_spa / (len(train_loader.dataset) / batch_size)
    train_loss_ent = train_loss_ent / (len(train_loader.dataset) / batch_size)

    return [train_loss_acc, train_loss_rec, train_loss_exp, train_loss_spa, train_loss_ent]



def save_model(weight_str, name):
    model_dict = {}
    model_dict['f_state_dict'] = f.state_dict()
    model_dict['g_state_dict'] = g.state_dict()
    model_dict['h_state_dict'] = h.state_dict()
    model_dict['d_state_dict'] = d.state_dict()
    model_dict['last_weights'] = weight_str
    model_dict['test_info'] = test_epoch_info
    model_dict['train_info'] = train_epoch_info
    torch.save('output/' + dataset + '_output/' + name, model_dict)
    return model_dict



#sample_data, sample_target, sample_locs = next(iter(test_loader))
sample_data, sample_target = next(iter(test_loader))

def sparse_sense(f, g, h, data, mults):
    f, g, h = f.eval(), g.eval(), h.eval()
    gdata = collect_g_data(f, g, h, device, data)[0]
    pred = h(torch.tensor(gdata).to(device)).argmax(dim=1).cpu().data.numpy()
    weights  = h.fc1.weight.cpu().data.numpy()
    result = []
    for multiplier in mults:
        sparse = 0
        for i in range(pred.shape[0]):
            expl_vec = gdata[i] * weights[pred[i]]
            thresh = expl_vec.max() / multiplier
            sparse += np.sum(expl_vec > thresh)
        print (sparse/pred.shape[0])
        result.append(sparse/pred.shape[0])
    return result

def sparse_h(h, gdata, data, thresh=0.1):
    weights = 1.0 * h.fc1.weight.data.numpy()
    weights[np.abs(weights) < thresh] = 0
    all_y = []
    for i in range(len(data.cat)):
        all_y += data.info[data.mode]['samp_per_class'] * [i]
    all_y = np.array(all_y)
    expl_out = gdata.dot(weights.T).argmax(axis=1)
    print (np.sum(expl_out == all_y) * 100.0 / 20000)
    return weights
    

def samp_category(f, g, h, test_data, thresh_f=0.1, thresh_e=0.07):
    f, g, h = f.eval(), g.eval(), h.eval()
    mat = np.zeros([2, 2, 2, 2])
    fuse_num = 0
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
    all_ent_f = []
    all_ent_e = []
    select_attr = []
    store_idx_sagR = []
    store_idx_agW = []
    store_idx_dag = []
    pred_corr = []
    pred_f = []
    pred_e = []
    orig_f = []
    orig_e = []
    batch_idx = -1
    for data in test_loader:
        batch_idx += 1
        inp, target = data[0].to(device), data[1].to(device)
        output, inter = f(inp)
        expl = h(g(inter))
        output, expl = output.softmax(dim=1), expl.softmax(dim=1)
        if batch_idx % 100 == 0:
            print (batch_idx)
        if batch_idx < 500 or batch_idx > 630:
            continue
        #fuse_p = (output + expl)/2
        #fuse_num += torch.sum(fuse_p.argmax(dim=1) == target)
        for i in range(16):
            ent_f = (-1 * output[i] * output[i].log()).sum()
            ent_e = (-1 * expl[i] * expl[i].log()).sum()
            all_ent_f.append(float(ent_f))
            all_ent_e.append(float(ent_e))
            #if (np.random.random() > 0.999):
                #print (output.min())
            idx1 = int(ent_f < thresh_f)
            idx2 = int(ent_e < thresh_e)
            idx3 = int(output.argmax(dim=1)[i] == expl.argmax(dim=1)[i])
            idx4 = int(output.argmax(dim=1)[i] == target[i])
            mat[idx1, idx2, idx3, idx4] += 1
            order = np.argsort(-1 * output[i].cpu().data.numpy())
            pred_f.append(output[i].cpu().data.numpy()[order])
            pred_e.append(expl[i].cpu().data.numpy()[order])
            orig_f.append(output[i].cpu().data.numpy())
            orig_e.append(expl[i].cpu().data.numpy())
            pred_corr.append(np.outer(output[i].cpu().data.numpy()[order], expl[i].cpu().data.numpy()[order]).flatten())
            #if idx1 == 1 and idx2 == 1 and idx3 == 0:
                #print (16 * batch_idx + i, int(target[i]), int(output.argmax(dim=1)[i]), int(expl.argmax(dim=1)[i]), output[i], expl[i])
            #if idx1 == 1 and idx2 == 1 and idx3 == 1 and idx4 == 1:
                #store_idx_sagR.append(16 * batch_idx + i)
            if idx3 == 0 and idx4 == 1 and int(target[i]) == 4:
                store_idx_sagR.append(int(expl.argmax(dim=1)[i]))
            if idx2 == 1 and idx3 == 0 and idx4 == 1 and int(target[i]) == 4:
                store_idx_agW.append(16 * batch_idx + i) 
            #elif idx3 == 1 and idx4 == 0:
                #store_idx_agW.append(16 * batch_idx + i)
            #elif idx3 == 0:
                #store_idx_dag.append(16 * batch_idx + i)
            #select_attr.append(g(inter)[i].cpu().data.numpy())
    #print (fuse_num)
    #return mat, np.array(select_attr), np.array(pred_corr), np.array(pred_f), np.array(pred_e), np.array(orig_f), np.array(orig_e)
    return store_idx_sagR, store_idx_agW, store_idx_dag


'''
for epoch in range(1, 12 + 1):
    test_info = test(f, g, h, d, device, test_loader, 1.0, 1.0, 1.0)
    train_info = train(f, g, h, d, device, train_loader, optimizer, epoch, lmbd_acc=1.0, lmbd_rec=0.8*int(epoch>0), lmbd_expl=0.5*int(epoch>2), lmbd_spa=0.1*int(epoch>2), lmbd_ent=0.2*int(epoch>4)) 
    test_epoch_info.append(test_info)
    train_epoch_info.append(train_info)

test_info = test(f, g, h, d, device, test_loader, 1.0, 1.0, 1.0)
test_epoch_info.append(test_info)
train_epoch_info = np.array(train_epoch_info)
test_epoch_info = np.array(test_epoch_info)
weight_str = 'lmbd_acc=1.0, lmbd_rec=5.0*int(epoch>0), lmbd_expl=0.1*int(epoch>2), lmbd_spa=0.02*int(epoch>1)+0.28*int(epoch>2), lmbd_ent=0.1*int(epoch>3)' # USE THIS FOR QUICKDRAW, ALSO CHANGE mu to 2.0
#weight_str = 'lmbd_acc=1.0, lmbd_rec=0.8*int(epoch>0), lmbd_expl=0.5*int(epoch>2), lmbd_spa=0.1*int(epoch>2), lmbd_ent=0.2*int(epoch>4)'

model_name = 'reference/sample'
model_dict = save_model(weight_str, 'sample_run', conciseness, model_name)
print ('Saved model', model_name)
'''


#checkpoint1 = torch.load('output/' + dataset + '_output/v18_if5_cd_0.30_run1.pt', map_location='cpu') # For QuickDraw
checkpoint1 = torch.load('output/' + dataset + '_output/model_opt_savedata_10.pt', map_location='cpu') # For MNIST
f.load_state_dict(checkpoint1['f_state_dict'])
g.load_state_dict(checkpoint1['g_state_dict'])
h.load_state_dict(checkpoint1['h_state_dict'])
d.load_state_dict(checkpoint1['d_state_dict'])
f, g, h, d = f.eval(), g.eval(), h.eval(), d.eval()

mults = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
conciseness = []
#data = datasets.FashionMNIST(home_dir + '/datasets', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor() ]))
#test_data = dl.QuickDraw(ncat=n_classes, mode='test')  
#test_data = datasets.CIFAR10(home_dir + '/datasets', train=False, download=True, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std) ]))


# EVALUATION

#facc, fidelity, gacc = analyze(f, g, h, d, device, test_loader, 'random/doen not matter')  

#conciseness = sparse_sense(f, g, h, test_data, mults)

# INTERPRETATION
print ('Generating explanations')
generate_model_explanations(f, g, h, d, train_data, device, dataset, model_name='sample_model', subset=True)

    


    






import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

criterion = nn.MSELoss()

def compute_saliency_maps(img, model):

    base = torch.ones(img.shape[0], img.shape[1], img.shape[2], img.shape[3])
    base = base.cuda()
    model.eval()
    grad_img = Variable(img.cuda(), requires_grad=True)

    output = model(grad_img)
    loss = criterion(grad_img, output)
    loss.backward()

    saliency = torch.abs(grad_img.grad)
    saliency, _ = torch.max(saliency, 1)
    saliency = saliency.data

    for i in range(len(base)):
        base[i][0] = torch.mul(base[i][0], saliency[i])
        base[i][1] = torch.mul(base[i][1], saliency[i])
        base[i][2] = torch.mul(base[i][2], saliency[i])

    return base


def normalise_saliency(saliency, prob):
    mid = torch.median(saliency)
    saliency = saliency.cpu().detach().numpy()
    for batch in range(0, len(saliency)):
        for x in range(len(saliency[batch])):
            for y in range(len(saliency[batch])):
                rand_var = random.randint(0, 100)
                if saliency[batch][x][y] < mid:
                    saliency[batch][x][y] = 1
                else:
                    if rand_var < (prob*100):
                        saliency[batch][x][y] = 0
                    else:
                        saliency[batch][x][y] = 1
    saliency = torch.tensor(saliency).cuda()
    return saliency


def show_saliency_maps(saliency, save, name):
    plt.imshow(saliency.cpu().detach().numpy(), cmap=plt.cm.hot)
    if save: plt.savefig(name + ".png")
    else: plt.show()


def show_tensor(tensor, save, name):
    tensor = tensor.cpu().detach().numpy()
    plt.imshow(np.transpose(tensor,(1,2,0)))
    if save: plt.savefig(name+".png")
    else: plt.show()


def apply_blackout(img, normalised):

    blackout_img = img.clone()
    blackout_img = blackout_img.cuda()
    normalised = normalised.cuda()

    # apply the blackout on the attention regions
    for i in range(len(blackout_img)):
        blackout_img[i][0] = torch.mul(normalised[i], blackout_img[i][0])
        blackout_img[i][1] = torch.mul(normalised[i], blackout_img[i][1])
        blackout_img[i][2] = torch.mul(normalised[i], blackout_img[i][2])

    return blackout_img

if __name__ == '__main__':
    pass
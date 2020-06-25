import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import numpy as np


criterion = nn.MSELoss()

def compute_saliency_maps(img, model):

    model.eval()
    grad_img = Variable(img.cuda(), requires_grad=True)

    output = model(grad_img)
    loss = criterion(grad_img, output)
    loss.backward()

    saliency = torch.abs(grad_img.grad)
    saliency, _ = torch.max(saliency, 1)
    saliency = saliency.data

    return saliency


def normalise_saliency(saliency):

    mid = torch.median(saliency)
    saliency = saliency.cpu().detach().numpy()
    for batch in range(0, len(saliency)):
        for x in range(len(saliency[batch])):
            for y in range(len(saliency[batch])):
                if saliency[batch][x][y] < mid:
                    saliency[batch][x][y] = 1
                else:
                    saliency[batch][x][y] = 0
    saliency = torch.tensor(saliency).cuda()
    return saliency


def show_saliency_maps(saliency):
    plt.imshow(saliency.cpu().detach().numpy(), cmap=plt.cm.hot)
    plt.show()


def show_tensor(tensor):
    tensor = tensor.cpu().detach().numpy()
    plt.imshow(np.transpose(tensor,(1,2,0)))
    plt.show()

if __name__ == '__main__':
    pass
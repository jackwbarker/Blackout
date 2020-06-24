import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


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

def show_saliency_maps(saliency):
    plt.imshow(saliency[0].cpu().detach().numpy(), cmap=plt.cm.binary)
    plt.show()

if __name__ == '__main__':
    pass
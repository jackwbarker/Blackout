import config, utilities
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch

from model import VQVAE

args = config.get_args()
transform = config.get_transform()

dataset = datasets.ImageFolder(args.path, transform=transform)
loader = DataLoader(dataset, batch_size=26, shuffle=True, num_workers=0)

model = VQVAE()
model = model.cuda()

criterion = nn.MSELoss()

from torch.autograd import Variable

for i, (img, label) in enumerate(loader) :
    #generate the attention regions for the images
    grad_img = Variable(img.cuda(), requires_grad=True)

    output = model(grad_img)
    loss = criterion(grad_img, output)
    loss.backward()


    saliency = torch.abs(grad_img.grad)
    saliency, _ = torch.max(saliency, 1)


    #apply the blackout on the attention regions




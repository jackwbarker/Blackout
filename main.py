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
    saliency = utilities.compute_saliency_maps(img, model)

    utilities.show_saliency_maps(saliency)
    break

    #apply the blackout on the attention regions




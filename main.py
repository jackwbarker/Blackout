import config, utilities
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch

from model import VQVAE

args = config.get_args()
transform = config.get_transform()

dataset = datasets.ImageFolder(args.path, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

model = VQVAE()
model = model.cuda()

criterion = nn.MSELoss()

from torch.autograd import Variable

for i, (img, label) in enumerate(loader) :

    #generate the attention regions for the images
    saliency = utilities.compute_saliency_maps(img, model)
    normalised = utilities.normalise_saliency(saliency)
    blackout_img = img.clone()

    utilities.show_tensor(blackout_img[0])

    blackout_img = blackout_img.cuda()

    #apply the blackout on the attention regions
    for i in range(len(blackout_img)):
        blackout_img[i][0] = torch.mul(normalised[0], blackout_img[i][0])
        blackout_img[i][1]=torch.mul(normalised[0], blackout_img[i][1])
        blackout_img[i][2]= torch.mul(normalised[0], blackout_img[i][2])

    utilities.show_saliency_maps(saliency[0])
    utilities.show_saliency_maps(normalised[0])
    utilities.show_tensor(blackout_img[0])


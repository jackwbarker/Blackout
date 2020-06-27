import config, utilities
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from tqdm import tqdm

from model import VQVAE

args = config.get_args()
transform = config.get_transform()

dataset = datasets.ImageFolder(args.path, transform=transform)
loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

model = VQVAE()
model = model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

from torch.autograd import Variable


for epoch in range(args.epoch):

    loader = tqdm(loader)
    
    for i, (img, _) in enumerate(loader) :
        img = img.cuda()

        #generate the attention regions for the images
        saliency = utilities.compute_saliency_maps(img, model)

        norm = transforms.Normalize((-1,-1,-1), (2,2,2))

        blackout_img = img.clone()

        for i in range(len(saliency)):
            normalised = norm(saliency[i])
            blackout_img[i] = torch.mul(normalised, img[i])


        #train model here
        model.train()
        model.zero_grad()
        output = model(blackout_img)
        loss = criterion(img, output)

        loss.backward()
        optimizer.step()

    print("EPOCH: ", epoch+1, "Loss: ", loss)
    torch.save(model.state_dict(), "Models/toy/"+str(epoch+1)+ ".pt")
    utilities.show_tensor(img[0], False, _)
    utilities.show_tensor(output[0], False, _)



import config
from torchvision import datasets
from torch.utils.data import DataLoader

args = config.get_args()
transform = config.get_transform()

dataset = datasets.ImageFolder(args.path, transform=transform)
loader = DataLoader(dataset, batch_size=26, shuffle=True, num_workers=0)

for i, (img, label) in enumerate(loader) :
    print(img)


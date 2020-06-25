import argparse
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=560)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--sched', type=str)
parser.add_argument('--device', type=str, default = 'cuda')
parser.add_argument('path', type=str)
args = parser.parse_args()

transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def get_args():
    return args

def get_transform():
    return transform
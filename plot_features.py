import os, argparse, torch, random
from os.path import  isfile, join
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import make_folder

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='svhn', choices=['svhn', 'cifar10'], help='Path to checkpoint file')
parser.add_argument('--data-path', type=str, default='./data', help='Data path')
parser.add_argument('--checkpoint-path', type=str, help='Path to checkpoint file')
parser.add_argument('--index-path', type=str, help='Path to indices file')
parser.add_argument('--save-path', type=str, help='Directory of save path')
parser.add_argument('--num-point', type=int, default=None, help='Number of points for unlabeled data and test data')
args = parser.parse_args()


class ConvLarge(nn.Module):
    def __init__(self, input_dim=3, num_classes=10, stochastic=True, top_bn=False):
        super(ConvLarge, self).__init__()
        self.block1 = self.conv_block(input_dim, 128, 3, 1, 1, 0.1)
        self.block2 = self.conv_block(128, 128, 3, 1, 1, 0.1)
        self.block3 = self.conv_block(128, 128, 3, 1, 1, 0.1)

        self.block4 = self.conv_block(128, 256, 3, 1, 1, 0.1)
        self.block5 = self.conv_block(256, 256, 3, 1, 1, 0.1)
        self.block6 = self.conv_block(256, 256, 3, 1, 1, 0.1)

        self.block7 = self.conv_block(256, 512, 3, 1, 0, 0.1)
        self.block8 = self.conv_block(512, 256, 1, 1, 0, 0.1)
        self.block9 = self.conv_block(256, 128, 1, 1, 0, 0.1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        maxpool = [nn.MaxPool2d(kernel_size=2, stride=2)]
        if stochastic: maxpool.append(nn.Dropout2d())
        self.maxpool = nn.Sequential(*maxpool)

        classifier = [nn.Linear(128, num_classes)]
        if top_bn: classifier.append(nn.BatchNorm1d(num_classes))
        self.classifier = nn.Sequential(*classifier)

    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=1, padding=1, lrelu_slope=0.01):
        return nn.Sequential(
                nn.Conv2d(input_dim, out_dim, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(inplace=True, negative_slope=lrelu_slope)
                )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.maxpool(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.maxpool(out)

        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)

        feature = self.avg_pool(out)
        feature = feature.view(feature.shape[0], -1)
        # logits = self.classifier(feature)

        return feature

meanstd = {
        'cifar10': [(0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)],
        'svhn': [(0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)]
        }

make_folder(args.save_path)

with open(args.index_path, 'r') as f:
    label_indices = [int(line.rstrip('\n')) for line in f]

data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*meanstd[args.dataset])
        ])
if args.dataset == "cifar10":
    train_dataset = dsets.CIFAR10(root=args.data_path, train=True, download=True, transform=data_transform)
elif args.dataset == "svhn":
    train_dataset = dsets.SVHN(root=args.data_path, split='train', download=True, transform=data_transform)

unlabel_indices = []
for idx in range(len(train_dataset.data)):
    if not idx in label_indices:
        unlabel_indices.append(idx)

label_num = len(label_indices)
unlabel_num = len(unlabel_indices)
print("# data: %d, labeled: %d, unlabeled: %d" % (len(train_dataset.data), label_num, unlabel_num))

label_indices = SubsetRandomSampler(label_indices)
unlabel_indices = SubsetRandomSampler(unlabel_indices)
label_loader = DataLoader(train_dataset, batch_size=100, num_workers=4, sampler=label_indices)
unlabel_loader = DataLoader(train_dataset, batch_size=100, num_workers=4, sampler=unlabel_indices)

if args.dataset == "cifar10":
    test_dataset = dsets.CIFAR10(root=args.data_path, train=False, download=True, transform=data_transform)
elif args.dataset == "svhn":
    test_dataset = dsets.SVHN(root=args.data_path, split='test', download=True, transform=data_transform)

test_loader = DataLoader(test_dataset, batch_size=100, num_workers=0)

model = ConvLarge(num_classes=10).cuda()

assert isfile(args.checkpoint_path), "No checkpoint at %s" % args.checkpoint_path
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()

label_features, unlabel_features, test_features = [], [], []
label_categories, unlabel_categories, test_categories = [], [], []

with torch.no_grad():
    print("Processing labeled data...")
    for step, (data, labels) in enumerate(label_loader):
        data = data.cuda()
        features = model(data)

        label_features.append(features)
        label_categories.append(labels)

        if step % 100 == 0:
            print("[%d/%d] Finished!" % (step, len(label_loader)))

    print("Processing unlabeled data...")
    for step, (data, labels) in enumerate(unlabel_loader):
        data = data.cuda()
        features = model(data)

        unlabel_features.append(features)
        unlabel_categories.append(labels)

        if step % 100 == 0:
            print("[%d/%d] Finished!" % (step, len(unlabel_loader)))

    print("Processing test data...")
    for step, (data, labels) in enumerate(test_loader):
        data = data.cuda()
        features = model(data)

        test_features.append(features)
        test_categories.append(labels)

        if step % 100 == 0:
            print("[%d/%d] Finished!" % (step, len(test_loader)))

label_features = torch.cat(label_features, dim=0).cpu().numpy()
unlabel_features = torch.cat(unlabel_features, dim=0).cpu().numpy()
test_features = torch.cat(test_features, dim=0).cpu().numpy()

label_categories = torch.cat(label_categories, dim=0).cpu().numpy()
unlabel_categories = torch.cat(unlabel_categories, dim=0).cpu().numpy()
test_categories = torch.cat(test_categories, dim=0).cpu().numpy()

all_features = np.concatenate([label_features, unlabel_features, test_features], axis=0)

print("Computing TSNE features...")
embedded_features = TSNE(n_components=2).fit_transform(all_features)
np.save(join(args.save_path, "aligned.npy"), embedded_features)
print("Done!")

embedded_label_features = embedded_features[:label_num, :]
embedded_unlabel_features = embedded_features[label_num:label_num+unlabel_num, :]
embedded_test_features = embedded_features[label_num+unlabel_num:, :]
if args.num_point is not None:
    idx = list(range(embedded_unlabel_features.shape[0]))
    random.shuffle(idx)
    idx = idx[:args.num_point]
    embedded_unlabel_features = embedded_unlabel_features[idx]
    unlabel_categories = unlabel_categories[idx]

    idx = list(range(embedded_test_features.shape[0]))
    random.shuffle(idx)
    idx = idx[:args.num_point]
    embedded_test_features = embedded_test_features[idx]
    test_categories = test_categories[idx]

color_dict = ["lightcoral", "darkgreen", "cadetblue", "darkorange", "peru",
              "cornflowerblue", "khaki", "magenta", "gray", "slategrey"]

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for i in range(len(color_dict)):
    idx_i = (label_categories == i)
    axes[0].scatter(embedded_label_features[idx_i, 0], embedded_label_features[idx_i, 1], c=color_dict[i], cmap=plt.cm.Spectral, s=10)
axes[0].set_xticks([])
axes[0].set_yticks([])

for i in range(len(color_dict)):
    idx_i = (unlabel_categories == i)
    axes[1].scatter(embedded_unlabel_features[idx_i, 0], embedded_unlabel_features[idx_i, 1], c=color_dict[i], cmap=plt.cm.Spectral, s=10)
axes[1].set_xticks([])
axes[1].set_yticks([])

for i in range(len(color_dict)):
    idx_i = (test_categories == i)
    axes[2].scatter(embedded_test_features[idx_i, 0], embedded_test_features[idx_i, 1], c=color_dict[i], cmap=plt.cm.Spectral, s=10)
axes[2].set_xticks([])
axes[2].set_yticks([])

axes[3].scatter(embedded_unlabel_features[:, 0], embedded_unlabel_features[:, 1], c="blue", cmap=plt.cm.Spectral, s=10)
axes[3].scatter(embedded_test_features[:, 0], embedded_test_features[:, 1], c="grey", cmap=plt.cm.Spectral, s=10)
axes[3].scatter(embedded_label_features[:, 0], embedded_label_features[:, 1], c="red", cmap=plt.cm.Spectral, s=10)
axes[3].set_xticks([])
axes[3].set_yticks([])

filename = join(args.save_path, "visualization.pdf")
plt.tight_layout()
plt.savefig(filename)
plt.close(fig)

# os.system("pdfcrop --margins \"0 0 0 0\" %s %s"%(filename,filename))


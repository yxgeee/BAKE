# Please download the raw data of TinyImageNet dataset from
# http://cs231n.stanford.edu/tiny-imagenet-200.zip

import os
import os.path as osp

root = '../'
ori = root+'tiny-imagenet-200/train'
tar = root+'tinyimagenet/train'
os.makedirs(tar)

for dir in os.listdir(ori):
    os.makedirs(osp.join(tar,dir))
    for file in os.listdir(osp.join(ori,dir,'images')):
        os.symlink(osp.join(ori,dir,'images',file), osp.join(tar,dir,file))


ori = root+'tiny-imagenet-200/val'
tar = root+'tinyimagenet/val'
os.makedirs(tar)

with open(root+'tiny-imagenet-200/val/val_annotations.txt','r') as f:
    list_file = f.readlines()

for item in list_file:
    item = item.strip()
    file = item.split('\t')[0]
    dir = item.split('\t')[1]
    if not osp.isdir(osp.join(tar,dir)):
        os.makedirs(osp.join(tar,dir))
    os.symlink(osp.join(ori,'images',file), osp.join(tar,dir,file))

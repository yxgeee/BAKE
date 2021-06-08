# Please download the raw data of MIT67 dataset from 
# http://web.mit.edu/torralba/www/indoor.html

import os
import os.path as osp
import scipy.io

root = '../'
ori = root+'Images'
tar = root+'MIT67'
os.makedirs(tar)

with open(root+'TrainImages.txt', 'r') as f:
    train_list = f.readlines()
with open(root+'TestImages.txt', 'r') as f:
    test_list = f.readlines()

def link(file_list, tar_folder):
    for name in file_list:
        name=name.strip()
        assert(osp.isfile(osp.join(ori,name)))
        if not osp.isdir(osp.dirname(osp.join(tar_folder, name))):
            os.makedirs(osp.dirname(osp.join(tar_folder, name)))
        os.symlink(osp.join(ori,name), osp.join(tar_folder, name))

link(train_list, osp.join(tar,'train'))
link(test_list, osp.join(tar,'test'))

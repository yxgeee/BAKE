# Please download the raw data of Stanford Dogs dataset from
# http://vision.stanford.edu/aditya86/ImageNetDogs/

import os
import os.path as osp
import scipy.io

root = '../'
ori = root+'Images'
tar = root+'STANFORD120'
os.makedirs(tar)

train_list = scipy.io.loadmat(root+'train_list.mat')['annotation_list']
test_list = scipy.io.loadmat(root+'test_list.mat')['annotation_list']

def link(file_list, tar_folder):
    for i in range(file_list.shape[0]):
        name = str(file_list[i][0][0])
        if not name.endswith('.jpg'):
            name = name+'.jpg'
        assert(osp.isfile(osp.join(ori,name)))
        if not osp.isdir(osp.dirname(osp.join(tar_folder, name))):
            os.makedirs(osp.dirname(osp.join(tar_folder, name)))
        os.symlink(osp.join(ori,name), osp.join(tar_folder, name))

link(train_list, osp.join(tar,'train'))
link(test_list, osp.join(tar,'test'))

# Please download the raw data of CUB_200_2011 dataset from
# https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view

import os
import os.path as osp

ori='../CUB_200_2011'
tar='../CUB200'
os.makedirs(tar)

with open(osp.join(ori, 'images.txt'), 'r') as f:
	img_list = f.readlines()
with open(osp.join(ori, 'train_test_split.txt'), 'r') as f:
	split_list = f.readlines()

img_dict = {}
for im in img_list:
	img_dict[im.split(' ')[0]]=im.strip().split(' ')[1]
split_dict = {}
for s in split_list:
	split_dict[s.split(' ')[0]]=int(s.strip().split(' ')[1])

for k in img_dict.keys():
	if split_dict[k]==1:
		s='train'
	else:
		s='test'
	if not os.path.isdir(os.path.join(tar,s,osp.dirname(img_dict[k]))):
		os.mkdir(os.path.join(tar,s,osp.dirname(img_dict[k])))
	os.symlink(osp.join(ori,'images',img_dict[k]),os.path.join(tar,s,img_dict[k]))

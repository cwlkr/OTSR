"""create hdf5 database for highspeed training"""
import torch
import tables
import re
import os,sys
import glob
import random

import PIL
import numpy as np

import cv2
import openslide
import os

import scipy.signal
import matplotlib.pyplot as plt

from openslide import OpenSlideUnsupportedFormatError
from sklearn import model_selection
import sklearn.feature_extraction.image

from tqdm.autonotebook import tqdm

dataname = "ovarian_multires"  # database name
pathdir = "/data/ovarian_wsi/"  # directory of wsi files
m_dir = 'masks/'  # directory of annatation masks saved in png


lvl = 1  # openslide level
patch_sizes = [64, 224]

set_seed = True
test_set_size = .2  # what percentage of the dataset should be used as a held out validation/testing set
downsampled = 16  # ds of annoations in m_dir
ext = '.mrxs'
project_re = f'(?=T).*(?<=HE)'  # regex to extract slide name

class_names = [["Tumor"], ["Stroma", 'Immune cells'], ['Necrosis', 'Other', 'adipose', 'vessels']]

# -- relating to masks
## different sample rates for tissues as well as cut-offs to include stroma regions smaller than 64px
cut_off_percent_ = {"Stroma": 0.33, 'Tumor': 0.9, 'Other': 0.9, 'Immune cells': 0.9, 'Necrosis': 0.9,
                    'adipose': 0.9, 'default':0.9}

max_number_samples_ = {"Stroma": 2500,  'Tumor': 2000, 'Other': 750, 'Immune cells': 750,
                    'Necrosis': 750, 'adipose': 200, 'default':500}

kernel_size = patch_sizes[0]//downsampled


def random_subset(a, b, nitems):
    assert len(a) == len(b)
    idx = np.random.randint(0, len(a), nitems)
    return a[idx], b[idx]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


seed = random.randrange(sys.maxsize)  # get a random seed so that we can reproducibly do the cross validation setup
if set_seed:
    seed = 119051062796297100
random.seed(seed)  # set the seed
print(f"random seed (note down for reproducibility): {seed}")
# -
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
filenameAtom = tables.StringAtom(itemsize=255)  # create an atom to store the filename of the image, just incase we need it later,
# +
files = glob.glob(f'./{m_dir}/*.png')


bases = [*set([re.search(project_re, f).group()for f in files])]

# create training and validation stages and split the files appropriately between them
phases = {}
phases["train"], phases["val"] = next(iter(model_selection.ShuffleSplit(n_splits=1,test_size=test_set_size).split(bases)))

storage = {}  # holder for future pytables

# block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
filters = tables.Filters(complevel=6, complib='zlib')  # we can also specify filters, such as compression, to improve storage speed

avg_filter = np.ones((kernel_size, kernel_size))

for phase in phases.keys():  # now for each of the phases, we'll loop through the files
    print(phase)

    totals = np.zeros(len(class_names))  # we can to keep counts of all the classes in for in particular training, since we
    if "/" != dataname[0]:
        dataname = f"./{dataname}"
    hdf5_file = tables.open_file(f"{dataname}_{lvl}_{phase}.pytable", mode='w')  # open the respective pytable
    storage["filenames"] = hdf5_file.create_earray(hdf5_file.root, 'filenames', filenameAtom, (0,)) #create the array for storage

    storage['seed'] = hdf5_file.create_carray(hdf5_file.root, 'seed', tables.Atom.from_dtype(np.asarray([seed]).dtype), np.asarray([seed]).shape)
    storage['seed'][:] = np.asarray(seed)
    for i, patch_size_ in enumerate(patch_sizes[-1]):  # we only use one patchsize here.
        storage[f"imgs_{i}"] = hdf5_file.create_earray(hdf5_file.root, f"imgs_{i}", img_dtype,
                                                  shape=np.append([0], np.array((patch_size_, patch_size_, 3)) ),
                                                  chunkshape=np.append([1], np.array((patch_size_, patch_size_, 3)) ),
                                                  filters=filters)
    storage["labels"] = hdf5_file.create_earray(hdf5_file.root, "labels", img_dtype,
                                                shape=[0],
                                                chunkshape=[1],
                                                filters=filters)
    for filei in phases[phase]:  # now for each of the files
        print(pathdir + bases[filei] + ext)
        mask_reg = f'(?<={bases[filei]}_).*(?=_mask)'
        try:
            osh = openslide.OpenSlide(pathdir + bases[filei] + ext)
        except openslide.OpenSlideUnsupportedFormatError:
            continue
        fname_masks = glob.glob(f'./{m_dir}/{bases[filei]}*.png')
        fname_mask = fname_masks[0]
        for fname_mask in fname_masks:

            print(fname_mask)
            mask_type = re.search(string=fname_mask, pattern=mask_reg).group()
            if not [c for c in class_names if mask_type in c]:
                print(f'{fname_mask} not used')
                continue
            for idx in range(len(class_names)):
                if mask_type in class_names[idx]:
                    classid = idx

            print(f'class label: {classid}')
            nr_samples = int(max_number_samples_[mask_type if mask_type in cut_off_percent_.keys() else 'default'])

            mask = cv2.imread(fname_mask)
            mask = mask[:, :, 0]
            mask = scipy.signal.fftconvolve(mask//255, avg_filter, mode='same')
            mask = mask >= (kernel_size**2)*cut_off_percent_[mask_type if mask_type in cut_off_percent_.keys() else 'default']

            avg_k = np.ones(((patch_sizes[1]//2)//downsampled, (patch_sizes[1]//2)//downsampled))
            a = (((patch_sizes[1]//2)//downsampled)**2)*0.9
            mask_ov_boarder = scipy.signal.fftconvolve(mask, avg_k, mode='same')
            mask_ov_boarder_ = np.logical_xor(mask, mask_ov_boarder >= a) # oversample boarder annoations at annoation boarders for small stromal regions.

            [rs, cs] = mask.nonzero()
            [rs_, cs_] = mask_ov_boarder_.astype(bool).nonzero()
            samples_ = int(min(nr_samples, len(rs))*0.7)
            samples1_ = min(nr_samples, len(rs)) - samples_
            [rs, cs] = random_subset(rs, cs, samples_)
            [rs_, cs_] = random_subset(rs, cs, samples1_)
            rs = np.concatenate((rs, rs_))
            cs = np.concatenate((cs, cs_))
            totals[classid] += len(rs)

            for i, (r, c) in tqdm(enumerate(zip(rs, cs)), total=len(rs)):
                for j, patch_size_ in enumerate(patch_sizes[-1]):  # saving only largest patchsize, we are croping smaller patch out of larger patch.
                    io = np.asarray(osh.read_region(((c*downsampled)-(patch_size_//2*2**lvl), (r*downsampled)-(patch_size_//2*2**lvl)), lvl, (patch_size_, patch_size_)))
                    io = io[:, :, 0:3]  # remove alpha channel
                    storage[f"imgs_{j}"].append(io[None, ::])
                storage["labels"].append([int(classid)])  # add the filename to the storage array
                storage["filenames"].append([fname_mask])  # add the filename to the storage array
        osh.close()
    # lastely, we should store the number of pixels
    npixels = hdf5_file.create_carray(hdf5_file.root, 'classsizes', tables.Atom.from_dtype(totals.dtype), totals.shape)
    npixels[:] = totals
    hdf5_file.close()

# +
# useful reference
# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

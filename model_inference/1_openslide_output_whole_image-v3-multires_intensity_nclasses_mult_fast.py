import os
import re
import glob
import cv2
import argparse

import openslide
import numpy as np
import sklearn.feature_extraction.image
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch.transforms import ToTensor

import model_training_inference.contextnet_loader as model_loader

class PatchDataset(Dataset):
    def __init__(self, data_set):
        "data set is output of sklearn extract_patches"
        self.data_set = data_set
        self.data_len = self.data_set.shape[0]*self.data_set.shape[1]
        self.shape = (self.data_set.shape[0], self.data_set.shape[1])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        i = np.unravel_index(indices=idx, shape=self.shape)
        return ToTensor()(image=self.data_set[i].squeeze())['image']


# +
parser = argparse.ArgumentParser(description='Make output for entire image using multi class contextawarenetwork ')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")
parser.add_argument('-p', '--patchsize', help="patchsize, default 64", default=64, type=int)
parser.add_argument('-p2', '--patchsize2', help="patchsize, default 224", default=224, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=128, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-n', '--num_workers', help="number of workers for loading data to gpu", default=8, type=float)
parser.add_argument('-l', '--openslide_level', help="openslide level, determines magnification level", default=1, type=float)
parser.add_argument('-u', '--mask_level', help="tissue mask level for skipping no tissue tiles", default=1, type=float)
parser.add_argument('-m', '--model', help="model", default="ovarian_multires.pth", type=str)
parser.add_argument('-a', '--class_name', help="post fix indicating which class was the target class", default="mult_classes", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)

# args = parser.parse_args(["-s", "256","-o", 'output_for_deeplab/', "-m", f'nki_ovarian_multires_T_SN_R_1305715_multires_best_model_freeze.pth', '/data/nki_ovarian_prelim/*.mrxs'])
args = parser.parse_args()
# checkpoint["slope"] = 0.2  # if not set in checkpoint

# +
print(args.model)
batch_size = args.batchsize
level = args.openslide_level
class_n = args.class_name
num_workers = args.num_workers
mask_level = args.mask_level
Path(args.outdir).mkdir(parents=True, exist_ok=True)

# +
model.eval()
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_properties(device))

model, name = model_loader.load_model(checkpoint, device=device)

# loading model state here because of apex
model.load_state_dict(checkpoint["model_dict"])
nclasses = model.classifier._modules['0'].out_channels
model.eval()

downsample = 2**level
offset = int(args.patchsize*downsample*0.5)

patch_size = args.patchsize
patch_size_2 = args.patchsize2
stride_size = patch_size // 4
tile_size = stride_size*64*2
tile_pad = patch_size_2-stride_size

if len(args.input_pattern) == 1:
    fnames = glob.glob(args.input_pattern[0])
elif not len(args.input_pattern):
    print('no files fond')
else:
    fnames = args.input_pattern
fname = fnames[0]

model = model.to(device)

# import ttach as tta
# model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(patch_size_2,patch_size_2))

for fname in fnames:

    name = os.path.splitext(os.path.basename(fname))[0]
    print(name)
    filename_out = f'{args.outdir}{name}_{class_n}.tif'
    if not args.force:
        if os.path.isfile(filename_out):
            continue
    try:
        osh = openslide.OpenSlide(fname)
        # add mask creation which skips parts of image
        img = osh.read_region((0, 0), mask_level, osh.level_dimensions[mask_level])
    except Exception as e:
        print(str(e) + " " + fname)
        continue

    img = np.asarray(img)[:, :, 0:3]
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.bitwise_and(imgg > 0, imgg < 250)

    del(img)
    del(use_mask)
    del(imgg)
    # +
    ds = int(osh.level_downsamples[level])
    shape = osh.level_dimensions[level]
    t_s_mask = int(tile_size*osh.level_downsamples[level]//2**mask_level)

    npmm = np.zeros((osh.level_dimensions[level][1]//stride_size, osh.level_dimensions[level][0]//stride_size, nclasses), dtype=np.uint8)
    for y in tqdm(range(0, osh.level_dimensions[0][1], round(tile_size * osh.level_downsamples[level])), desc="outer"):
        for x in range(0, osh.level_dimensions[0][0], round(tile_size * osh.level_downsamples[level])):

            maskx = int(x//osh.level_downsamples[mask_level])
            masky = int(y//osh.level_downsamples[mask_level])

            if(maskx >= mask.shape[1] or masky >= mask.shape[0]) or mask[masky:masky+t_s_mask,maskx:maskx+t_s_mask].mean() < 0.05: #need to handle rounding error
                continue

            xr, yr = x-offset, y-offset

            if (xr < 0) or (x+(tile_size+tile_pad)*downsample >= osh.level_dimensions[0][0]):
                continue
            if (yr < 0) or (y+(tile_size+tile_pad)*downsample >= osh.level_dimensions[0][1]):
                continue

            io = np.asarray(osh.read_region((xr-((patch_size_2-patch_size)//2), yr-((patch_size_2-patch_size)//2)), level, (tile_size+tile_pad,tile_size+tile_pad)))[:,:,0:3] #trim alpha

            output = np.zeros((0, nclasses, patch_size//patch_size, patch_size//patch_size))
            arr_out = sklearn.feature_extraction.image._extract_patches(io, (patch_size_2, patch_size_2, 3), stride_size)
            arr_out_shape = arr_out.shape
            dataLoader = DataLoader(PatchDataset(arr_out), batch_size=batch_size, pin_memory = True, num_workers=num_workers, shuffle=False)
            for batch_arr in dataLoader:
                batch_arr_gpu_2 = batch_arr.to(device)
                batch_arr_gpu_1 = batch_arr_gpu_2[:, :, (patch_size_2//2) - (patch_size//2):(patch_size_2//2) + (patch_size//2),(patch_size_2//2) - (patch_size//2):(patch_size_2//2) + (patch_size//2)]
                # ---- get results
                with torch.no_grad():
                    output_batch = model(batch_arr_gpu_1, batch_arr_gpu_2)  #first 64 then 224
                    output_batch_color = F.softmax(output_batch, dim=1).cpu().numpy()

                output = np.append(output, output_batch_color[:, :, None, None], axis=0)

            output = output.transpose((0, 2, 3, 1))
            # turn from a single list into a matrix of tiles
            output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size//patch_size, patch_size//patch_size, output.shape[3])

            # turn all the tiles into an image
            output = np.concatenate(np.concatenate(output, 1), 1)

            cut_to_dim = npmm[y//stride_size//ds:y//stride_size//ds+tile_size//stride_size,x//stride_size//ds:x//stride_size//ds+tile_size//stride_size,:].shape
            npmm[y//stride_size//ds:y//stride_size//ds+tile_size//stride_size, x//stride_size//ds:x//stride_size//ds+tile_size//stride_size, :] = output[0:cut_to_dim[0], 0:cut_to_dim[1],:]*255 #need to save uint8
    # -
    from tifffile import TiffWriter
    with TiffWriter(filename_out, bigtiff=True, imagej=True) as tif:
        tif.save(npmm, compress=6, tile=(256, 256))

import glob
import tifffile
import cv2
import openslide
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import pool
import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description='Convert bigtiff to png annotation masks for downstream computations')

parser.add_argument('-o', '--outdir', help="outputdir, default ./output_bin/", default="./output_bin/", type=str)
parser.add_argument('-d', '--downsample', help="downsample for saving annoation masks", default=16, type=float)
parser.add_argument('-x', '--extension', help="wsi extension", default=".mrxs", type=str)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False, action="store_true")
parser.add_argument('wsipath', help="path to WSI", default="/data/ovarian_wsi/", type=str)
parser.add_argument('path', help="path tissue annoation bigtiff from inference script",
                                                    default="output", type=str)
args = parser.parse_args()

wsi_path = args.wsipath
force = args.force
outdir = args.outdir
ext =args.extension
ds = args.downsample
ouput_key = ['0', 'tumor', 'stroma', 'bg']

files = glob.glob(os.path.join(args.path, '*.tif'))
Path(outdir).mkdir(parents=True, exist_ok=True)

def process_file(file):
    # name = re.search('T.*HE', file).group()
    name = os.path.splitext(os.path.basename(file))[0]

    if not force:
        if os.path.exists(f'{outdir}/{name}_{ouput_key[1]}.png'):
            return
    try:
        slide = openslide.open_slide(f'{os.path.join(wsi_path, name)}{ext}')
    except openslide.OpenSlideError:
        return
    mask_level = slide.get_best_level_for_downsample(ds + 0.5)
    shape_dim = slide.level_dimensions[mask_level]
    img = tifffile.imread(file)/255
    img = np.concatenate((np.zeros(img.shape[0:2])[:, :, None], img), axis=2)
    img_A = np.argmax(img, axis=2)
    img_A = img_A[0:shape_dim[1],0:shape_dim[0]]
    imgg = cv2.cvtColor(np.array(slide.read_region((0, 0), mask_level, shape_dim))[:, :, 0:3], cv2.COLOR_RGB2GRAY)/255
    for i in range(1, img.shape[2]):
        # set all white to zero
        sub = (img_A == i)*1
        sub = cv2.resize(sub.astype(np.uint8), dsize=shape_dim, interpolation=cv2.INTER_NEAREST)

        l = np.zeros(sub.shape)
        l[64//32:, 64//32:] = sub[:-64//32, :-64//32]  # add patch offset.
        cv2.imwrite(f'{outdir}/{name}_{ouput_key[i]}.png', (l*255).astype(np.uint8))


with pool.Pool(6) as p:
    r = list(tqdm.tqdm(p.imap(process_file, files), total=len(files)))

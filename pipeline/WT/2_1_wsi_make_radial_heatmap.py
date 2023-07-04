import cv2
import glob
import re
import argparse
import os
import tqdm

import numpy as np
import openslide
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.signal import fftconvolve
from skimage.morphology import disk
from multiprocessing import pool


parser = argparse.ArgumentParser(description='Create radial heatmap for WT tumor bed estimation')

parser.add_argument('-o', '--outdir', help="outputdir, default ./intensities_dif_kernel/", default="./intensities_dif_kernel", type=str)
parser.add_argument('-d', '--downsample', help="downsample for calculation", default=16, type=float)
parser.add_argument('-x', '--extension', help="wsi extension", default=".mrxs", type=str)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('wsipath',
                    help="path to whole slide images",
                    default="/data/ovarian_wsi/", type=str)
parser.add_argument('path',
                    help="path tissue annoation bigtiff from inference script",
                    default="output_bin", type=str)
args = parser.parse_args()

outdir = args.outdir
im_p = args.path
wsi_path=args.wsipath
ext = args.extension
force = args.force
ds = args.downsample # downsample for output

rmm = 0.75

bg = 'bg'
tumor = 'tumor'
stroma = 'stroma'

def split_m_n(file):
    return os.path.basename(file).split(f'_{stroma}')[0].split(f'_{bg}')[0].split(f'_{tumor}')[0]

def load_and_resize(file, im_p, tissue, slide_size):
    map = cv2.imread(f'{os.path.join(im_p, file)}_{tissue}.png')
    if map is None:
        return
    return cv2.resize(map[:,:,0]//255, dsize = slide_size, interpolation=cv2.INTER_NEAREST)

def calc_tsr(file):
    if os.path.exists(os.path.join(outdir, f'{file}_heatmap_{rmm}.png')) and not force:
        print(f'skipping {file}')
        return
    # print(f'----- start analyzing {file} -----')
    try:
        slide = openslide.open_slide(f'{os.path.join(wsi_path, file)}{ext}')
        mask_level=slide.get_best_level_for_downsample(ds+0.5)
        slide_size = slide.level_dimensions[mask_level]
        uppx = slide.properties['openslide.mpp-y']
        img_gray = cv2.cvtColor(np.asarray(slide.read_region((0,0),mask_level,slide_size))[:,:,0:3], cv2.COLOR_RGB2GRAY)/255
    except Exception as e:
        print(e, file)
        return

    # img_gray = np.logical_not((img_gray>0.925)*1 + (img_gray ==0)*1)
    # img_bg =  load_and_resize(file, im_p, bg, slide_size)
    uppx = float(uppx)
    r_at_40 = (1/uppx)*1000*rmm
    radius = int(np.floor(r_at_40/slide.level_downsamples[mask_level]))  #
    kernel = disk(np.floor(radius))

    # img gray is tumor+stroma only.
    # img_gray_ = img_gray *1 # - np.logical_and(img_gray.astype(np.bool), img_bg.astype(np.bool))
    del(img_gray)
    del(img_bg)
    ## t_map
    t_map = load_and_resize(file, im_p, tumor, slide_size)
    t_map_conv = fftconvolve(t_map*1, kernel, mode= 'same')
    del(t_map)

    ## s_map
    s_map = load_and_resize(file, im_p, stroma, slide_size)
    s_map_conv = fftconvolve(s_map*1, kernel, mode= 'same')
    del(s_map)

    img_gray_conv = t_map_conv+s_map_conv
    ## fft introduces small < 0 values; set to 0
    s_map_conv[s_map_conv<0] = 0
    t_map_conv[t_map_conv<0] = 0
    img_gray_conv[img_gray_conv<0] = 0

    tmp = t_map_conv + s_map_conv
    tmp[(t_map_conv + s_map_conv) <= 0.001] = -1  # remove where t_map and s_map both zero set to - to have distinct values
    res = t_map_conv/(tmp) # tumor_perc
    res[res <= 0 ] = 0  # remove where t_map and s_map both zero or negative due to setting denominator to negative values
    ### remove tissue edges
    res[(img_gray_conv) < 0.33 * np.sum(kernel)] = 0
    cv2.imwrite(os.path.join(outdir, f'{file}_heatmap_{rmm}.png'), res*255)

if __name__ == "__main__:

    files_raw = glob.glob(f'{im_p}/*')
    files = [*set([split_m_n(i) for i in files_raw])]

    Path(outdir).mkdir(parents=True, exist_ok=True)

    with pool.Pool(1) as p:
        r = list(tqdm.tqdm(p.imap(calc_tsr, files), total=len(files)))

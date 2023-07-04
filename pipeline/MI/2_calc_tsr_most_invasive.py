import cv2
from scipy.signal import fftconvolve
import glob
import re
from skimage.morphology import disk
import numpy as np
import openslide
import matplotlib.pyplot as plt
import os
import tqdm
import pandas as pd
from scipy.stats import describe
from pathlib import Path
import argparse
from multiprocessing import pool

parser = argparse.ArgumentParser(description='Calculate MI-TSR')

parser.add_argument('-d', '--downsample', help="downsample for calculation", default=16, type=float)
parser.add_argument('-x', '--extension', help="wsi extension", default=".mrxs", type=str)
parser.add_argument('wsipath',
                    help="path of whole slide images",
                    default="/data/ovarian_wsi/", type=str)
parser.add_argument('path',
                    help="path to tissue annoations in png format per tissue from inference script",
                    default="output_bin", type=str)
args = parser.parse_args()

im_p = args.path
wsi_path = args.wsipath
ext =args.extension
mask_ds = args.downsample # downsample for calculation

bg = 'bg'  # if there is aliasing
tumor = 'tumor'
stroma = 'stroma'

def calc_tsr(file):
    try:
        slide = openslide.open_slide(f'{os.path.join(wsi_path, file)}{ext}')
    except:
        print(file, 'slide error')
        return
    mask_level = slide.get_best_level_for_downsample(mask_ds+0.5)  # ds for calculation. Calculate in 16ds from 40x, 0.5 error offset of openslide
    slide_size = slide.level_dimensions[mask_level]

    uppx = slide.properties['openslide.mpp-y']
    uppx = float(uppx)
    r_at_40 = (1/uppx)*1000*1.6
    radius = int(np.floor(r_at_40/mask_ds))  # 16x ds
    kernel = disk(np.floor(radius))

    img_gray = cv2.cvtColor(np.asarray(slide.read_region((0,0),mask_level,slide_size))[:,:,0:3], cv2.COLOR_RGB2GRAY)/255
    img_gray = np.logical_not((img_gray>0.925)*1 + (img_gray ==0)*1)
    img_bg = cv2.imread(f'{im_p}{file}_{bg}.png')[:,:,0]
    img_bg = cv2.resize(img_bg, dsize = slide_size, interpolation=cv2.INTER_NEAREST)

    del(img_gray)
    del(img_bg)
    ## t_map
    t_map = cv2.imread(f'{im_p}{file}_{tumor}.png')[:,:,0]//255
    t_map = cv2.resize(t_map, dsize = slide_size, interpolation=cv2.INTER_NEAREST)
    t_map = np.logical_and(t_map,t) if tumor_bed_mode is not None else t_map
    t_map_conv = fftconvolve(t_map*1, kernel, mode= 'same')
    del(t_map)

    ## s_map
    s_map = cv2.imread(f'{im_p}{file}_{stroma}.png')[:,:,0]//255
    s_map = cv2.resize(s_map, dsize = slide_size, interpolation=cv2.INTER_NEAREST)
    s_map = np.logical_and(s_map,t) if tumor_bed_mode is not None else s_map
    s_map_conv = fftconvolve(s_map*1, kernel, mode= 'same')
    del(s_map)

    img_gray_conv = t_map_conv+s_map_conv  # combined tissue of interest in kernel
    ## fft introduces small < 0 values; set to 0
    s_map_conv[s_map_conv<0] = 0
    t_map_conv[t_map_conv<0] = 0
    img_gray_conv[img_gray_conv<0] = 0

    tmp = t_map_conv + s_map_conv
    tmp[(t_map_conv + s_map_conv) <= 0.001] = -1  # remove where t_map and s_map both zero set to - to have distinct values
    res = t_map_conv/(tmp)

    res[res <= 0 ] = 0  # remove where t_map and s_map both zero or negative due to setting denominator to negative values
    # cv2.imwrite(f'intensities/{file}heatmap.png', res*255)
    res[(img_gray_conv) < 0.66*np.sum(kernel)] = 0  # only measure in regions with adequat tissue present.

    most_invasive_t = np.unravel_index(np.argmax(res), res.shape)

    most_tsr = s_map_conv[most_invasive_t]/tmp[most_invasive_t]

    return {'slide': file, 'tsr_mi':most_tsr}


def split_m_n(file):
    return os.path.basename(file).split(f'_{stroma}')[0].split(f'_{bg}')[0].split(f'_{tumor}')[0]

if __name__ == '__main__':

    files_raw = glob.glob(f'{im_p}/*')

    files = [*set([split_m_n(i) for i in files_raw])]
    with pool.Pool(20) as p:
        r = list(tqdm.tqdm(p.imap(calc_tsr, files), total=len(files)))
    #r = [calc_tsr(file) for file in tqdm.tqdm(files)]

    def list_dict_to_dict_list(ldict):
        ldict = [i for i in ldict if i is not None]
        d = {i:[] for i in ldict[0].keys()}
        [d[j[0]].append(j[1]) for i in ldict  if i is not None for j in i.items()]
        return d
    # Analysis Block
    Path("results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list_dict_to_dict_list(r)).to_csv(f'results/tsr_radial_{tumor_bed_mode}_3_2.csv')

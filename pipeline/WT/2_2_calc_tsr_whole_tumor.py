import cv2
import glob
import re
import numpy as np
import openslide
import matplotlib.pyplot as plt
import os
import tqdm
import sys
import pandas as pd
from pathlib import Path
from multiprocessing import pool
from multiprocessing import Lock
import argparse

parser = argparse.ArgumentParser(description='Calculate WT-TSR')

parser.add_argument('-d', '--downsample', help="downsample for calculation", default=16, type=float)
parser.add_argument('-x', '--extension', help="wsi extension", default=".mrxs", type=str)
parser.add_argument('-h', '--heatmappath',
                    help="path tissue annoation bigtiff from inference script",
                    default="intensities_dif_kernel", type=str)
parser.add_argument('wsipath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="/data/ovarian_wsi/", type=str)
parser.add_argument('path',
                    help="path tissue annoation bigtiff from inference script",
                    default="output_bin", type=str)
args = parser.parse_args()

bg = 'bg'
tumor = 'tumor'
stroma = 'stroma'

t_ = 0.075  # threshold for tumor percentage in t_bed

radial_heatmap_path = args.heatmappath
im_p = args.path

ext=args.extension
wsi_path = args.wsipath
ds = args.downsample  # in what downsample are we calculating the scores.

def split_m_n(file):
    return os.path.basename(file).split(f'_{stroma}')[0].split(f'_{bg}')[0].split(f'_{tumor}')[0]


def radial_t_bed(file, slide_size, mask):
    global t_
    global radial_heatmap_path
    m_radial = cv2.imread(f'{os.path.join(radial_heatmap_path, file)}_heatmap_0.75.png')
    m_radial = cv2.resize(m_radial, (imgg.shape[1], imgg.shape[0]), interpolation = cv2.INTER_NEAREST)
    m_pred = (m_radial/255 > t_) * 1
    t = np.logical_and(m_pred, mask[:,:, None])
    t__ = t_
    while np.any(t.astype(bool)) == False:  # iterative shrinking for small tumors
        t__ -= 0.05
        if t__ <= 0:
            tqdm.tqdm.write(f'{file} is heamap empty?',file=sys.stdout)
            break
        m_pred = (m_radial/255 > t__) * 1
        t = np.logical_and(m_pred, mask[:,:, None])
    return t

def load_and_resize(file, im_p, tissue, slide_size):
    map = cv2.imread(f'{os.path.join(im_p, file)}_{tissue}.png')
    if map is None:
        return
    return cv2.resize(map[:,:,0]//255, dsize = slide_size, interpolation=cv2.INTER_NEAREST)


def glob_tsr(file):
    global tumor_bed_mode
    global name
    t = None
    slide = openslide.open_slide(os.path.join(wsi_path, f'{file}{ext}'))
    mask_level = slide.get_best_level_for_downsample(ds+0.5)
    slide_size = slide.level_dimensions[mask_level]

    # print(file)
    img = slide.read_region((0, 0), mask_level, slide_size)
    img = np.asarray(img)[:, :, 0:3]
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.bitwise_and(imgg > 0, imgg < 250)
    t = radial_t_bed(file, slide_size, mask)

    if t is None or np.any(t.astype(bool)) is False:
        print(file, 'error in t bed')
        return

    t = cv2.resize(t[:,:,0].astype(np.uint8), dsize = slide_size, interpolation=cv2.INTER_NEAREST)
    img_gray = cv2.cvtColor(np.asarray(slide.read_region((0,0),mask_level,slide_size))[:,:,0:3], cv2.COLOR_RGB2GRAY)/255
    img_gray = np.logical_not((img_gray>0.925)*1 + (img_gray ==0)*1)

    img_bg = load_and_resize(file, im_p, bg, slide_size)

    if img_bg is None:  #  no files found of this slide.
        return

    # img gray is tumor+stroma only.
    img_gray_ = img_gray *1 - np.logical_and(img_gray.astype(bool), img_bg.astype(bool))
    del(img_gray)
    del(img_bg)
    # t_map
    t_map = load_and_resize(file, im_p, tumor, slide_size)
    # s_map
    s_map = load_and_resize(file, im_p, stroma, slide_size)

    # only measure in tumor bed
    s_map = np.logical_and(s_map.astype(bool), t.astype(bool))
    t_map = np.logical_and(t_map.astype(bool), t.astype(bool))
    tsr = np.sum(s_map)/(np.sum(s_map) + np.sum(t_map)+0.000000000000000000000001)

    frame = pd.DataFrame({'slide': file, 'tsr_total': tsr}, index = [1])
    with csv_lock:
        if os.path.exists(name):
            frame.to_csv(name, mode='a', header=False, index=False)
        else:
            frame.to_csv(name, mode='w', index=False)
    return {'slide': file, 'tsr_total':tsr}



if __name__ == '__main__':
    files_raw = glob.glob(f'{im_p}*')
    files = [*set([split_m_n(i) for i in files_raw])]
    Path("results").mkdir(parents=True, exist_ok=True)

    name = f'results/tsr_wsi_{tumor_bed_mode}_075kernel_t1_2_dt0_05.csv'
    if os.path.exists(name):
        os.remove(name)

    csv_lock = Lock()
    with pool.Pool(10) as p:
        r = list(tqdm.tqdm(p.imap(glob_tsr, files), total=len(files), file=sys.stdout))

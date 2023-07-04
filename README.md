# Pipeline for calculating MI-TSR and WT-TSR and ContextNet model training and inference


## Model Training

1. Create a hdf5 database for highspeed deep learning training on annoations masks

```
python 1_make_hdf5_classification_multires.py
```
2. Train ContextNet based on parameters in config.json file.

```
python 2_train_multires_classification.py
```

## Model inference

1. Create a hdf5 database for highspeed deep learning training on annoations masks

```
python 1_openslide_output_whole_image-v3-multires_intensity_nclasses_mult_fast.py
```
2. Create a hdf5 database for highspeed deep learning training on annoations masks

```
python 1_1_tiff_to_mask.py
```

## Scoring Pipeline

The scoring pipeline uses WSI tissue masks annoations from the trained ContextNet.
WSI scoring scripts can be used directly if WSI level tissue masks are provided for tumor, stroma and bg in the correct naming convention: f'{slidename}_{tissuename}.png'

1. Create WSI tissue annoations from trained ContextNet.
2. Calculate TSR scores based on tissue WSI annotations


### MI-TSR


To calculate scores for MI-TSR slides with WSI tissue annoations run in specified folder run:

```
python 2_calc_tsr_most_invasive.py
```

### WT-TSR

To calculate scores for MI-TSR slides with WSI tissue annoations run in specified folder run:

```
python 2_1_wsi_make_radial_heatmap.py
```
to initalize radial tumor heatmaps for the tumor bad estmation and subsequently run:

```
python 2_2_calc_tsr_whole_tumor.py
```

to calculate scores WT-TSR for all slides with WSI tissue annoations run in specified folder.

"""training script for context aware network"""

import time
import math
import tables
import json
import sys
import scipy.sparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as albmt
from albumentations.pytorch import ToTensor
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import ttach as tta
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import random
import imgaug
# 05/11/2019

module_list = ['contextnet']


# helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class Dataset(object):

    def __init__(self, fname, img_transform=None):
        self.fname = fname
        self.img_transform = img_transform

        with tables.open_file(self.fname, 'r') as db:
            self.classsizes = db.root.classsizes[:]
            self.nitems = db.root.imgs_0.shape[0]

        self.imgs0 = None
        self.imgs2 = None
        self.labels = None


    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

        with tables.open_file(self.fname, 'r') as db:
            self.imgs0 = db.root.imgs_0
            self.labels = db.root.labels

            # get the requested image and mask from the pytable
            img0 = self.imgs0[index, :, :, :]
            img2 = self.imgs0[index, 80:224-80, 80:224-80, :]

            label = self.labels[index]

        img_new0 = img0
        img_new2 = img2

        if self.img_transform:
            # update seed because both transforms should do the same transformations!!

            img_new0 = self.img_transform[0](image=img0)['image']
            img_new2 = self.img_transform[1](image=img2)['image']

        # return img_new0, img_new1, img_new2, label
        return img_new0, img_new2, label

    def __len__(self):
        return self.nitems

def get_dataloader(dataname, phases, batch_size, num_workers):
    phase = 'val'
    if "/" != dataname[0]:
        dataname = f"./{dataname}"
    dataset = {}
    dataLoader = {}
    for phase in phases:  # now for each of the phases, we're creating the dataloader
        # interestingly, given the batch size, i've not seen any improvements from using a num_workers>0

        dataset[phase] = Dataset(f"{dataname}_{phase}.pytable", img_transform=(img_transform, img_transform1))
        drop_last = True if len(dataset[phase]) % batch_size == 1 else False
        dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size, drop_last = drop_last,
                                       shuffle=True, num_workers=num_workers, pin_memory=True)
        print(f"{phase} dataset size:\t{len(dataset[phase])}")
    nclasses = dataset["train"].classsizes.shape[0]
    class_weight = dataset["train"].classsizes
    # visualize a single example to verify that it is correct
    (img,img1, label) = dataset["train"][10]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # build output showing patch after augmentation and original patch
    ax[0].imshow(np.moveaxis(img.numpy(), 0, -1))
    ax[1].imshow(np.moveaxis(img1.numpy(), 0, -1))
    print(img.shape)
    print(img1.shape)
    #print(label)
    # -
    return dataLoader, nclasses, class_weight


# +
def trainnetwork(num_epochs, optim, criterion, num_classes, phases, validation_phases, conf, device , dataLoader, checkpointing, model_name, checkpoint=None):


    writer = SummaryWriter()  # open the tensorboard visualiser
    best_loss_on_test = np.Infinity if checkpoint is None else checkpoint['best_loss_on_test']
    start_time = time.time()
    start_epoch = 0 if conf['retrain'] is False else checkpoint['epoch']
    for epoch in range(start_epoch, num_epochs):
        # zero out epoch based performance variables
        alpha = 1
        all_acc = {key: 0 for key in phases}
        all_loss = {key: torch.zeros(0).to(device) for key in phases}  # keep this on GPU for greatly improved performance
        cmatrix = {key: np.zeros((num_classes, num_classes)) for key in phases}

        for phase in phases:  # iterate through botfh training and validation states

            if phase == 'train':
                model.train()  # Set model to training mode
            else:  # when in eval mode, we don't want parameters to be updated
                model.eval()   # Set model to evaluate mode

            tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(conf['patch_size'],conf['patch_size']))
            for ii, (X, Y, label) in enumerate(dataLoader[phase]):
                # for each of the batches
                X = X.to(device)  # [Nbatch, 3, H, W]
                Y = Y.to(device)  # [Nbatch, 3, H, W]

                if conf['target_class'] is not None:
                    label = (label == conf['target_class']).type('torch.LongTensor').to(device)  # [Nbatch, 1] with
                else:
                    label = label.type('torch.LongTensor').to(device)  # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)

                with torch.set_grad_enabled(phase == 'train'):

                    if phase == "train":
                        prediction = model(X, Y)  # [N, Nclass]
                    else:
                        prediction = tta_model(X, Y)
                    loss = criterion(prediction, label)

                    if phase == "train":  # in case we're in train mode, need to do back propogation
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                    all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1,-1)))

                    if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                        p = prediction.detach().cpu().numpy()
                        cpredflat = np.argmax(p,axis=1).flatten()
                        yflat = label.cpu().numpy().flatten()
                        CM = scipy.sparse.coo_matrix((np.ones(yflat.shape[0], dtype=np.int64), (yflat, cpredflat)),
                            shape=(conf["num_classes"], conf["num_classes"]), dtype=np.int64,
                            ).toarray()
                        cmatrix[phase] = cmatrix[phase]+CM

                        all_acc[phase] = (cmatrix[phase]/cmatrix[phase].sum()).trace()
            all_loss[phase] = all_loss[phase].cpu().numpy().mean()

            # save metrics to tensorboard
            writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
            if phase in validation_phases:
                writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
                for r in range(num_classes):
                    for c in range(num_classes):  # essentially write out confusion matrix
                        writer.add_scalar(f'{phase}/{r}{c}', cmatrix[phase][r][c], epoch)

        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs),
                                                                       epoch+1, num_epochs, (epoch+1) / num_epochs * 100,
                                                                       all_loss["train"], all_loss["val"]),end="")

        # if current loss is the best we've seen, save model state with all variables
        # necessary for recreation
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **")
            save_model(epoch, model_name, conf["dataname"], model, optim, conf, best_loss_on_test)
        else:
            print("")

    return model, optim, all_loss

def save_model(epoch, model_name, dataname, model, optim, conf, test_loss, desc="best"):
    state = {'epoch': epoch + 1,
             'model_dict': model.state_dict(),
             'optim_dict': optim.state_dict(),
             'best_loss_on_test': test_loss,
             'growth_rate': conf["growth_rate"],
             'block_config': conf['block_config'],
             'num_init_features': conf["num_init_features"],
             'bn_size': conf["bn_size"],
             'slope' : conf['slope'],
             'l2' : conf['l2'],
             'drop_rate': conf["drop_rate"],
             'num_classes': conf["num_classes"],
             'width_factor': conf['width_factor']}

    torch.save(state, f"{dataname}_{model_name}_{desc}_model.pth")


def load_config(path):
    with open(path) as file:
        str = file.read()
    conf = json.loads(str)
    conf['block_config'] = conf['block_config']
    return conf


if __name__ == '__main__':

    conf = load_config("model_training_inference/config.json")
    module = conf['module']
    # if argparse. Let it overrule config file by writing in conf dict
    print(conf)
    model, device, model_name = None, None, None

    assert module in module_list
    device = torch.device(conf['gpuid'])
    loader = None
    exec(f'import model_training_inference.{module}_loader as loader')
    model, model_name = loader.load_model(conf, device)


    checkpoint = None
    n_params = sum([np.prod(p.size()) for p in model.parameters()])

    if conf['retrain']:

        model_path_n = conf['dataname'] + '' if conf["target_class"] is None else f'_t{conf["target_class"]}'
        model_path_n +=  f'{n_params}'
        checkpoint = torch.load(f"{model_path_n}_{model_name}_last_model.pth", map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to dev
        checkpoint['best_loss_on_test'] = torch.load(f"{model_path_n}_{model_name}_best_model.pth", map_location=lambda storage, loc: storage)['best_loss_on_test'] #load checkpoint to CPU and then put to dev
        print("continue training network after finished run")

    patch_size = conf['patch_size']

    img_transform = albmt.Compose([
        albmt.RandomScale(scale_limit=0.1,p=.9),
        albmt.PadIfNeeded(min_height=patch_size,min_width=patch_size),
        albmt.VerticalFlip(p=.5),
        albmt.HorizontalFlip(p=.5),
        albmt.Blur(p=.2),
        # albmt.Downscale(p=.25, scale_min=0.64, scale_max=0.99),
        albmt.GaussNoise(p=.33, var_limit=(10.0, 50.0)),
        # albmt.GridDistortion(p=.5, num_steps=5, distort_limit=(-0.3, 0.3),
        #                border_mode=cv2.BORDER_REFLECT),
        albmt.ISONoise(p=.33, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        albmt.RandomBrightness(p=.5, limit=(-0.2, 0.2)),
        albmt.RandomContrast(p=.5, limit=(-0.2, 0.2)),
        albmt.RandomGamma(p=.5, gamma_limit=(80, 120), eps=1e-07),
        albmt.MultiplicativeNoise(p=.5, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
        albmt.HueSaturationValue(hue_shift_limit=15,sat_shift_limit=10,val_shift_limit=10,p=.9),
        albmt.Rotate(p=1, border_mode=cv2.BORDER_REFLECT),
        albmt.RandomCrop(patch_size, patch_size),
        ToTensor()
    ])
    img_transform1 = albmt.Compose([
        albmt.RandomScale(scale_limit=0.1,p=.9),
        albmt.PadIfNeeded(min_height=224, min_width=224),
        albmt.VerticalFlip(p=.5),
        albmt.HorizontalFlip(p=.5),
        albmt.Blur(p=.2),
        # albmt.Downscale(p=.25, scale_min=0.64, scale_max=0.99),
        albmt.GaussNoise(p=.33, var_limit=(10.0, 50.0)),
        # albmt.GridDistortion(p=.5, num_steps=5, distort_limit=(-0.3, 0.3),
        #                border_mode=cv2.BORDER_REFLECT),
        albmt.ISONoise(p=.33, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        albmt.RandomBrightness(p=.5, limit=(-0.2, 0.2)),
        albmt.RandomContrast(p=.5, limit=(-0.2, 0.2)),
        albmt.RandomGamma(p=.5, gamma_limit=(80, 120), eps=1e-07),
        albmt.MultiplicativeNoise(p=.5, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
        albmt.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=10, val_shift_limit=10, p=.9),
        albmt.Rotate(p=1, border_mode=cv2.BORDER_REFLECT),
        albmt.RandomCrop(224, 224),
        ToTensor()
    ])
    optim = torch.optim.Adam(model.parameters())  # adam is going to be the most robust, though perhaps not the best performing, typically a good place to start


    if conf['retrain']:

        model.load_state_dict(checkpoint['model_dict'])
        optim.load_state_dict(checkpoint['optim_dict'])

    # we have the ability to weight individual classes, in this case we'll do so based on their presense in the trainingset
    # to avoid biasing any particular class
    dataLoader, nclasses, class_weight = get_dataloader(conf["dataname"], conf["phases"], conf["batch_size"], conf['num_workers'])
    if conf['target_class'] is not None:
        conf['dataname'] = conf['dataname'] + f'_t{conf["target_class"]}'
        w = class_weight[conf['target_class']]/class_weight.sum()
        class_weight = torch.from_numpy(np.asarray([w, 1-w])).type('torch.FloatTensor').to(device)
    else:
        class_weight = torch.from_numpy(1-class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)
    conf['dataname'] = conf['dataname'] + f'{n_params}'

    print(class_weight)  # show final used weights, make sure that they're reasonable before continouing
    # -

    criterion = nn.CrossEntropyLoss(weight=class_weight)

    (model, optim, all_loss) = trainnetwork(conf["num_epochs"], optim, criterion, conf["num_classes"],
                                            conf["phases"], conf["validation_phases"],
                                            conf, device, dataLoader, conf['checkpointing'], model_name,
                                            checkpoint)
    save_model(conf["num_epochs"]-1, model_name, conf["dataname"], model, optim, conf, all_loss['val'], desc="last")

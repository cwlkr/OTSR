from model_training_inference.multiresmodel import ContextNet
import numpy as np


def load_model(conf, device=None, load=False):
    """ conf is python dict containing """
    # +
    # build the model according to the paramters specified above. finally print out the number of trainable parameters
    try:
        conf['l2']
    except KeyError:
        conf['l2'] = 0
    try:
        conf['slope']
    except KeyError:
        conf['slope'] = 0
    model = ContextNet(conf['num_classes'], width_factor=conf['width_factor'], l2=conf['l2'], slope = conf['slope'])
    print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

    if device is not None:
        model = model.to(device)

    if load is True:
        model.load_state_dict(conf['model_dict'])
        print('Model loaded')
    return model, 'multires'

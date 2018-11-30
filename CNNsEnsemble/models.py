from nnmodels import *
import numpy as np
from senet import senet154
from inceptionv4 import inceptionv4

def create_model(model_config):
    model_name = model_config.name
    n_classes = model_config.getint("n_classes")
    model = None

    # Layer that duplicate 1-channel image to a 3-channels image
    new_first_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True)]

    if model_name.startswith("senet"):
        model = senet154(num_classes=n_classes, pretrained=None)
        new_first_layer.extend(list(model.layer0))
        model.layer0 = nn.Sequential(*new_first_layer)

    elif model_name.startswith("inception"):
        model = inceptionv4(num_classes=n_classes, pretrained=None)
        new_first_layer.extend(list(model.features))
        model.features = nn.Sequential(*new_first_layer)

    return model
from torchvision.transforms import *
import torch
from denoisingTransformRaw import remove_noise

noisy_mean = 0.0916588985098
noisy_std = 0.256155366338

denoised_mean = 0.0129291835294
denoised_std = 0.10511452131

class FiveCropStacked(FiveCrop):
    def __call__(self, img):
        crops = super(FiveCropStacked, self).__call__(img)
        return torch.stack([(ToTensor()(crop)) for crop in crops]).squeeze()

class Denoise(object):
    def __call__(self, img):
        assert(img.shape[-1] == 1)
        img = remove_noise(img.squeeze())
        return img[:,:, None]

    def __repr__(self):
        return self.__class__.__name__

def transformation_constructor(transformations_list, model_config):
    transformations_names = transformations_list.split(",")
    transforms = []

    if model_config.getboolean("noisy"):
        mean = noisy_mean
        std = noisy_std
    else:
        mean = denoised_mean
        std = denoised_std

    mean = [mean] * model_config.getint("n_dim")
    std = [std] * model_config.getint("n_dim")

    for transformation_name in transformations_names:
        transformation = None

        if transformation_name == "RandomRotation":
            transformation = RandomRotation(model_config.getfloat("rotation_angle"))

        elif transformation_name == "RandomTranslation":
            transformation = RandomAffine(translate=model_config.getfloat("translation"))

        elif transformation_name == "RandomScale":
            transformation = RandomAffine(scale=model_config.getfloat("scale"))

        elif transformation_name == "RandomHorizontalFlip":
            transformation = RandomHorizontalFlip()

        elif transformation_name == "FiveCropStacked":
            transformation = FiveCropStacked(model_config.getint("five_crop"))

        elif transformation_name == "Resize":
            transformation = Resize(model_config.getfloat("resize"))

        elif transformation_name == "ToPILImage":
            transformation = ToPILImage()

        elif transformation_name == "ToTensor":
            transformation = ToTensor()

        elif transformation_name == "Normalize":
            transformation = Normalize(mean=mean, std=std)

        elif transformation_name == "Denoise":
            transformation = Denoise()

        elif transformation_name == "RescaleToInputSize":
            transformation = Resize(model_config.getint("img_scale"))

        if transformation is not None:
            transforms.append(transformation)

    return Compose(transforms)

def create_transforms(model_config):
    assert(model_config.get("train_transforms") != None)
    train_transforms = transformation_constructor(model_config.get("train_transforms"), model_config)

    assert(model_config.get("valid_transforms") != None)
    valid_transforms = transformation_constructor(model_config.get("valid_transforms"), model_config)

    if model_config.get("test_transforms"):
        test_transforms = transformation_constructor(model_config.get("test_transforms"), model_config)
    else:
        test_transforms = valid_transforms

    return {"train": train_transforms, "valid": valid_transforms, "test": test_transforms}
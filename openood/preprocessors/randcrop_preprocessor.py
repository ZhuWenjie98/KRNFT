import torch
import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict


class RandCropPreprocessor():
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]
        #normalization_type = config.dataset.normalization_type
        #if normalization_type in normalization_dict.keys():
        #    self.mean = normalization_dict[normalization_type][0]
        #    self.std = normalization_dict[normalization_type][1]
        #else:
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.n_crop = config.preprocessor.n_crop
        self.random_crop = tvs_trans.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            tvs_trans.RandomResizedCrop(224),
            tvs_trans.RandomHorizontalFlip(),
            tvs_trans.ToTensor(),
            tvs_trans.Normalize(mean=self.mean, std=self.std)
        ])
        

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        views = [self.random_crop(image).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        return views

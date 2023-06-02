import numpy as np
import torchvision.transforms.functional as F
import torch
import PIL


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms):
        """
        Inputs
            files: list
                Paths to each image file.
            transforms: Transformations Class Object
                Torchvision image transformations.
        """
        self.files = files
        self.transforms = transforms

    def __getitem__(self, i):
        # Load image from the hard disc.
        img = PIL.ImageOps.grayscale(PIL.Image.open(self.files[i]))

        # Apply any transforms to the data if required.
        if self.transforms is not None:
            img = self.transforms(img)

        ret = {'img': img, 'img_name': self.files[i]}
        return ret

    def __len__(self):
        return len(self.files)


def collate_double(batch):
    """
    collate function for the ImageDataset.
    Only used by the dataloader.
    """
    x = [sample['img'] for sample in batch]
    y = [sample['img_name'] for sample in batch]
    return x, y


class Transformations:
    """
    Instantiates the required image transformations for the image dataset 
    (uses standard transforms from the pretrained model via the 'model_weights' argument)
    """

    def __init__(self, model_weights):
        """
        model_weights: the weights object (i.e. FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        """
        self.transforms = self.get_transformations(model_weights)

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def get_transformations(self, weights):
        transforms = []
        transforms.append(lambda x: self.toTensor(x))
        transforms.append(lambda x: weights.transforms()(x))

        return transforms

    def toTensor(self, image):
        """
        Converts a PIL image to a tensor
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image).type(torch.float32)
        else:
            image = F.pil_to_tensor(image).type(torch.float32)
        return image

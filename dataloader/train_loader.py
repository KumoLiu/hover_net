import csv
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
class FileLoader(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        file_list,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.mode, seed)
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # split stacked channel into image and label
        img = (data[..., :3]).astype("uint8")  # RGB images
        ann = (data[..., 3:]).astype("int32")  # instance ID map and type map

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW1 -> HW
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: document hard coded assumption about #input
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
                iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -A to +A percent (per axis)
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # shear by -5 to +5 degrees
                    rotate=(-179, 179),  # rotate by -179 to +179 degrees
                    order=0,  # use nearest neighbour
                    backend="cv2",  # opencv for fast processing
                    seed=rng,
                ),
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),
                iaa.Flipud(0.5, seed=rng),
            ]

            input_augs = [
                iaa.OneOf(
                    [
                        # iaa.Lambda(
                        #     seed=rng,
                        #     func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        # ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # set position to 'center' for center crop
                # else 'uniform' for random crop
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs


from monai.data import Dataset, DataLoader
from monai.transforms import *
from skimage import measure
from monai.utils import set_determinism, convert_to_tensor, convert_to_numpy
import sys
sys.path.append('/home/yunliu/Workspace/Code/tutorials/pathology/hovernet')
from transforms import RandShiftHued, RandShiftSaturationd

def prepare_data(files):
    images, labels, inst_maps, type_maps = [], [], [], []
    for file in files:
        data = np.load(file)
        images.append(data[..., :3].transpose(2, 0, 1))
        inst_maps.append(measure.label(data[..., 3][None]).astype(int))
        type_maps.append(data[..., 4][None])
        labels.append(np.array(data[..., 3][None] > 0, dtype=int))

    data_dicts = [
        {"img": _image, "np_map": _label, "label_inst": _inst_map, "tp_map": _type_map}
        for _image, _label, _inst_map, _type_map in zip(images, labels, inst_maps, type_maps)
    ]

    return data_dicts


from scipy.ndimage import measurements
from skimage import morphology as morph

def original_hovermap_transform(ann, crop_shape=(80, 80)):
    def get_bounding_box(img):
        """Get bounding box coordinate information."""
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # due to python indexing, need to add 1 to max
        # else accessing will be 1px in the box, not out
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]

    def cropping_center(x, crop_shape, batch=False):
        orig_shape = x.shape
        if not batch:
            h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
            x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        else:
            h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
            w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
            x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
        return x

    def fix_mirror_padding(ann):
        current_max_id = np.amax(ann)
        inst_list = list(np.unique(ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(ann == inst_id, np.uint8)
            remapped_ids = measurements.label(inst_map)[0]
            remapped_ids[remapped_ids > 1] += current_max_id
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann

    def gen_instance_hv_map(ann, crop_shape):

        ann = convert_to_numpy(ann)
        orig_ann = ann.copy()  # instance ID map
        fixed_ann = fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, crop_shape)
        # TODO: deal with 1 label warning
        crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

        x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
        y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

        inst_list = list(np.unique(crop_ann))
        inst_list.remove(0)  # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(fixed_ann == inst_id, np.uint8)
            inst_box = get_bounding_box(inst_map)

            # expand the box by 2px
            # Because we first pad the ann at line 207, the bboxes
            # will remain valid after expansion
            inst_box[0] -= 2
            inst_box[2] -= 2
            inst_box[1] += 2
            inst_box[3] += 2

            inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

            if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
                continue

            # instance center of mass, rounded to nearest pixel
            inst_com = list(measurements.center_of_mass(inst_map))

            inst_com[0] = int(inst_com[0] + 0.5)
            inst_com[1] = int(inst_com[1] + 0.5)

            inst_x_range = np.arange(1, inst_map.shape[1] + 1)
            inst_y_range = np.arange(1, inst_map.shape[0] + 1)
            # shifting center of pixels grid to instance center of mass
            inst_x_range -= inst_com[1]
            inst_y_range -= inst_com[0]

            inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

            # remove coord outside of instance
            inst_x[inst_map == 0] = 0
            inst_y[inst_map == 0] = 0
            inst_x = inst_x.astype("float32")
            inst_y = inst_y.astype("float32")

            # normalize min into -1 scale
            if np.min(inst_x) < 0:
                inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
            if np.min(inst_y) < 0:
                inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
            # normalize max into +1 scale
            if np.max(inst_x) > 0:
                inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
            if np.max(inst_y) > 0:
                inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

            ####
            x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            x_map_box[inst_map > 0] = inst_x[inst_map > 0]

            y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
            y_map_box[inst_map > 0] = inst_y[inst_map > 0]

        hv_map = np.dstack([x_map, y_map])
        return hv_map, crop_ann

    """Generate the targets for the network."""
    hv_map, crop_ann = gen_instance_hv_map(ann.squeeze(), crop_shape)

    hv_map = cropping_center(hv_map, crop_shape)
    return hv_map.transpose(2, 0, 1)


def get_monai_dataset(data_list, phase):
    if phase == "train":
        transforms = Compose(
            [
                RandAffined(
                    keys=["img", "label_inst", "tp_map"],
                    prob=1.0,
                    rotate_range=((np.pi), 0),
                    scale_range=((0.2), (0.2)),
                    shear_range=((0.1), (0.1)),
                    translate_range=((6), (6)),
                    padding_mode="zeros",
                    mode=("nearest"),
                        ),
                CenterSpatialCropd(
                    keys="img", 
                    roi_size=(270, 270),
                ),
                RandFlipd(keys=["img", "label_inst", "tp_map"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["img", "label_inst", "tp_map"], prob=0.5, spatial_axis=1),
                OneOf(transforms=[
                    # RandGaussianSmoothd(keys=["img"], sigma_x=(0.5,1.15), sigma_y=(0.5,1.15), prob=1.0),
                    MedianSmoothd(keys=["img"], radius=2),
                    RandGaussianNoised(keys=["img"], prob=1.0, std=0.05)
                ]),
                Compose(
                [
                    RandShiftHued(keys=["img"], offsets=(-8, 8), prob=1.0),
                    RandShiftSaturationd(keys=["img"], offsets=(-0.2, 0.2), clip=True, prob=1.0),
                    RandShiftIntensityd(keys=["img"], offsets=(-26, 26), clip=True, prob=1.0),
                    RandAdjustContrastd(keys=["img"], prob=1.0, gamma=(0.75,1.25))], shuffle=True),
                #! AsDiscreted(keys=["tp_map"], to_onehot=[5]),
                # ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                CastToTyped(keys="label_inst", dtype=torch.int),
                # ComputeHoVerMapsd(keys="label_inst"),
                Lambdad(keys='label_inst', func=original_hovermap_transform, overwrite=False, new_key='hv_map'),
                Lambdad(keys="label_inst", func=lambda x: x>0, overwrite=False, new_key='np_map'),
                #! AsDiscreted(keys=["np_map"], to_onehot=2),
                CenterSpatialCropd(
                    keys=["np_map", "label_inst", "tp_map", "hv_map"], 
                    roi_size=(80, 80),
                ),
                CastToTyped(keys=["img", "label_inst", "tp_map"], dtype=torch.float32),
            ]
        )
    elif phase == "valid":
        transforms = Compose(
            [
                CenterSpatialCropd(
                    keys="img", 
                    roi_size=(270, 270),
                ),
                # ScaleIntensityRanged(keys=["img"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                CastToTyped(keys="label_inst", dtype=torch.int),
                # ComputeHoVerMapsd(keys="label_inst"),
                Lambdad(keys='label_inst', func=original_hovermap_transform, overwrite=False, new_key='hv_map'),
                Lambdad(keys="label_inst", func=lambda x: x>0, overwrite=False, new_key='np_map'),
                CenterSpatialCropd(
                    keys=["np_map", "label_inst", "tp_map", "hv_map"], 
                    roi_size=(80, 80),
                ),
                CastToTyped(keys=["img", "label_inst", "tp_map"], dtype=torch.float32),
            ]
        )
    else:
        raise NotImplementedError(f"got {phase}")
    
    numpy_data_list = prepare_data(data_list)

    print("data list:", len(numpy_data_list))

    return Dataset(data=numpy_data_list, transform=transforms)

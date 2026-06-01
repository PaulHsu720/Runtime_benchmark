# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
Supports both PIL Image and NumPy array inputs.
"""
import os
import random
from typing import Union, Dict, Any, Tuple, List, Optional

import PIL
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

try:
    from groundingdino.util.box_ops import box_xyxy_to_cxcywh
    from groundingdino.util.misc import interpolate
except ImportError:
    # Fallback if not in groundingdino environment
    def box_xyxy_to_cxcywh(boxes):
        """Convert box format from xyxy to cxcywh"""
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)

    def interpolate(mask, size, mode="nearest"):
        """Simple interpolation fallback"""
        import torch.nn.functional as F
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).float()
            result = F.interpolate(mask, size=size, mode=mode)
            return result.squeeze(1) > 0.5
        return F.interpolate(mask.unsqueeze(0).float(), size=size, mode=mode).squeeze(0) > 0.5


# =============================================================================
# Image Type Utilities
# =============================================================================
ImageType = Union[np.ndarray, PIL.Image.Image, torch.Tensor]
TargetType = Optional[Dict[str, Any]]


def is_numpy_image(img: ImageType) -> bool:
    """Check if input is numpy array (image)"""
    return isinstance(img, np.ndarray)


def is_pil_image(img: ImageType) -> bool:
    """Check if input is PIL Image"""
    return isinstance(img, PIL.Image.Image)


def is_tensor_image(img: ImageType) -> bool:
    """Check if input is torch Tensor"""
    return isinstance(img, torch.Tensor)


def to_pil_image(img: ImageType) -> PIL.Image.Image:
    """Convert numpy array or tensor to PIL Image"""
    if is_pil_image(img):
        return img
    elif is_tensor_image(img):
        # Tensor to PIL
        if img.dim() == 3 and img.shape[0] in [1, 3, 4]:
            # CHW format
            img = img.cpu().detach()
            # Normalize to [0, 1] if needed
            if img.max() > 1:
                img = img / 255.0
            img = img.permute(1, 2, 0)  # CHW -> HWC
        elif img.dim() == 2:
            img = img.cpu().detach()
        img_np = img.numpy()
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        return PIL.Image.fromarray(img_np)
    elif is_numpy_image(img):
        # Numpy array to PIL
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return PIL.Image.fromarray(img)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def to_tensor(img: ImageType) -> torch.Tensor:
    """Convert numpy array or PIL Image to torch Tensor (CHW format)"""
    if is_tensor_image(img):
        return img
    elif is_pil_image(img):
        return F.to_tensor(img)
    elif is_numpy_image(img):
        # HWC numpy -> CHW tensor
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        # Convert HWC -> CHW
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")


def to_numpy(img: ImageType, dtype=np.uint8) -> np.ndarray:
    """Convert PIL Image or tensor to numpy array"""
    if is_numpy_image(img):
        if img.dtype == dtype:
            return img
        if dtype == np.uint8:
            return (img * 255).clip(0, 255).astype(np.uint8)
        return img.astype(dtype)

    if is_tensor_image(img):
        # CHW -> HWC
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img_np = img.cpu().detach().numpy()
        if dtype == np.uint8:
            if img_np.max() <= 1:
                img_np = (img_np * 255).clip(0, 255)
            return img_np.astype(np.uint8)
        return img_np.astype(dtype)

    if is_pil_image(img):
        img_np = np.array(img)
        if dtype == np.uint8 and img_np.dtype != np.uint8:
            return img_np.astype(np.uint8)
        return img_np

    raise TypeError(f"Unsupported image type: {type(img)}")


# =============================================================================
# Transform Functions (支持 numpy/PIL/Tensor)
# =============================================================================
def crop(image: ImageType, target: TargetType, region: Tuple[int, int, int, int]):
    """Crop image and update target boxes/masks"""
    i, j, h, w = region

    if is_numpy_image(image):
        # NumPy: direct slicing (faster)
        cropped_image = image[i:i+h, j:j+w]
    else:
        # PIL or Tensor: use functional
        cropped_image = F.crop(to_pil_image(image), *region)

    target = target.copy() if target else {}
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "positive_map"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, i:i+h, j:j+w]
        fields.append("masks")

    # Remove elements with zero area boxes or masks
    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    if os.environ.get("IPDB_SHILONG_DEBUG", None) == "INFO":
        if "strings_positive" in target:
            target["strings_positive"] = [
                _i for _i, _j in zip(target["strings_positive"], keep) if _j
            ]

    return cropped_image, target


def hflip(image: ImageType, target: TargetType) -> Tuple[ImageType, TargetType]:
    """Horizontal flip"""
    if is_numpy_image(image):
        # NumPy: direct flip
        flipped_image = np.fliplr(image).copy()
    else:
        # PIL or Tensor
        flipped_image = F.hflip(to_pil_image(image))

    target = target.copy() if target else {}
    w, h = get_image_size(image)

    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def get_image_size(img: ImageType) -> Tuple[int, int]:
    """Get image (width, height)"""
    if is_numpy_image(img):
        return img.shape[1], img.shape[0]  # (w, h)
    elif is_tensor_image(img):
        if img.dim() == 3:
            return img.shape[2], img.shape[1]  # CHW -> (w, h)
        return img.shape[1], img.shape[0]
    elif is_pil_image(img):
        return img.size  # PIL returns (w, h)
    raise TypeError(f"Unsupported image type: {type(img)}")


def resize(image: ImageType, target: TargetType, size, max_size=None):
    """Resize image and scale boxes"""
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    w, h = get_image_size(image)
    size_wh = get_size((w, h), size, max_size)  # Returns (h, w)
    target_h, target_w = size_wh

    if is_numpy_image(image):
        # NumPy: use cv2 resize
        import cv2
        if image.ndim == 3:
            rescaled_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            rescaled_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        # PIL or Tensor
        rescaled_image = F.resize(to_pil_image(image), size_wh)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip([target_w, target_h], [w, h]))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    target["size"] = torch.tensor([target_h, target_w])

    if "masks" in target:
        target["masks"] = interpolate(
            target["masks"][:, None].float(), (target_h, target_w), mode="nearest"
        )[:, 0] > 0.5

    return rescaled_image, target


def pad(image: ImageType, target: TargetType, padding: Tuple[int, int]):
    """Pad image on bottom-right corners"""
    pad_x, pad_y = padding

    if is_numpy_image(image):
        # NumPy: direct padding
        if image.ndim == 3:
            padded_image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
        else:
            padded_image = np.pad(image, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    else:
        padded_image = F.pad(to_pil_image(image), (0, 0, pad_x, pad_y))

    if target is None:
        return padded_image, None

    target = target.copy()
    if is_numpy_image(padded_image):
        target["size"] = torch.tensor([padded_image.shape[0], padded_image.shape[1]])
    else:
        target["size"] = torch.tensor(padded_image.size[::-1])

    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(
            target["masks"], (0, pad_x, 0, pad_y)
        )

    return padded_image, target


# =============================================================================
# Transform Classes
# =============================================================================
class ResizeDebug:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(to_pil_image(img), self.size)
        return crop(img, target, region)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes

    def __call__(self, img: ImageType, target: Dict):
        init_boxes = len(target["boxes"])
        max_patience = 10
        for i in range(max_patience):
            pil_img = to_pil_image(img)
            w = random.randint(self.min_size, min(pil_img.width, self.max_size))
            h = random.randint(self.min_size, min(pil_img.height, self.max_size))
            region = T.RandomCrop.get_params(pil_img, [h, w])
            result_img, result_target = crop(img, target, region)
            if (
                not self.respect_boxes
                or len(result_target["boxes"]) == init_boxes
                or i == max_patience - 1
            ):
                return result_img, result_target
        return result_img, result_target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        w, h = get_image_size(img)
        crop_height, crop_width = self.size
        crop_top = int(round((h - crop_height) / 2.0))
        crop_left = int(round((w - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad:
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect:
    """Randomly selects between transforms1 and transforms2"""
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    """Convert image to tensor (CHW format)"""
    def __call__(self, img, target):
        return to_tensor(img), target


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        if is_tensor_image(img):
            return self.eraser(img), target
        return self.eraser(to_tensor(img)), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # Ensure tensor input
        if not is_tensor_image(image):
            image = to_tensor(image)

        image = F.normalize(image, mean=self.mean, std=self.std)

        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[-2:]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes

        return image, target


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# =============================================================================
# Output Types (for returning consistent formats)
# =============================================================================
class OutputMode:
    """Output mode for transforms"""
    NUMPY = "numpy"
    PIL = "pil"
    TENSOR = "tensor"
    SAME = "same"  # Keep as input type


class MultiModeCompose(Compose):
    """Compose that supports multiple output modes"""
    def __init__(self, transforms: List, output_mode: str = OutputMode.SAME):
        super().__init__(transforms)
        self.output_mode = output_mode

    def convert_output(self, image, target):
        """Convert output to desired format"""
        if self.output_mode == OutputMode.TENSOR:
            return to_tensor(image), target
        elif self.output_mode == OutputMode.PIL:
            return to_pil_image(image), target
        elif self.output_mode == OutputMode.NUMPY:
            return to_numpy(image), target
        return image, target  # SAME mode

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return self.convert_output(image, target)


# =============================================================================
# Convenience Functions
# =============================================================================
def apply_transforms(
    image: ImageType,
    target: TargetType,
    transform_list: List,
    output_mode: str = OutputMode.SAME
) -> Tuple[ImageType, TargetType]:
    """Apply list of transforms to image and target"""
    composed = MultiModeCompose(transform_list, output_mode)
    return composed(image, target)


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Test with numpy
    import cv2

    # Create test image (numpy)
    test_image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create test target
    test_target = {
        "boxes": torch.tensor([[100, 100, 200, 200], [50, 50, 150, 150]]),
        "labels": torch.tensor([1, 2]),
        "area": torch.tensor([10000, 10000])
    }

    # Test transforms
    transforms = Compose([
        RandomHorizontalFlip(p=1.0),
        RandomResize([800], max_size=1333),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply to numpy
    result_np, target = transforms(test_image_np, test_target)
    print(f"Input type: {type(test_image_np)} -> Output type: {type(result_np)}")
    print(f"Output shape: {result_np.shape}")
    print(f"Target boxes: {target['boxes']}")

    # Apply to PIL
    test_image_pil = PIL.Image.fromarray(test_image_np)
    result_pil, _ = transforms(test_image_pil, test_target.copy())
    print(f"\nPIL Input -> Output type: {type(result_pil)}")
    print(f"Output shape: {result_pil.shape}")

    print("\n✓ All transforms support numpy, PIL, and tensor inputs!")
import re
from pathlib import Path

import nibabel
from scipy.interpolate import interp1d
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
from dpipe.im.box import mask2bounding_box
from dpipe.io import load_json
from dpipe.im.preprocessing import get_greatest_component, normalize
from dpipe.im import crop_to_box
from dpipe.im.shape_ops import zoom
from dpipe.im.utils import apply_along_axes

NII_PATTERN = re.compile(r'(.*)\.nii(\.gz)?')
PIXEL_SPACING = np.array([0.5, 0.5])


def gather_nifty(folder):
    folder = Path(folder)
    files = folder.rglob('*') if folder.is_dir() else [folder]
    for file in sorted(files):
        if not file.is_file():
            continue

        match = NII_PATTERN.fullmatch(file.name)
        if match:
            yield match.group(1), file


def gather_train(folder):
    folder = Path(folder)

    paths = {}
    for name, image in gather_nifty(folder):
        contours = image.with_name(f'{name}.json')
        if contours.exists():
            paths[str(image)] = (str(image), str(contours))

    return list(paths.values())


def load_pair(image_path, contours_path):
    image = nibabel.load(image_path)
    contours = load_json(contours_path)

    x = image.get_fdata()
    # interpolate a list of lists of points to a simple numpy array
    contours = np.array([
        np.stack(list(interpolate_contours(c, x.shape[0])), -1) for c in contours
    ], dtype='float32')

    # interpolation to PIXEL_SPACING
    scale_factor = image.header.get_zooms()[:2] / PIXEL_SPACING
    x = remove_background(zoom(x, scale_factor, axis=[0, 1]))
    contours = zoom(contours, scale_factor[0], axis=1) * scale_factor[1]

    # cropping the redundant zero padding that might be in the image
    start, stop = box = mask2bounding_box(x != 0)
    x = normalize_image(crop_to_box(x, box))
    contours = crop_to_box(contours, box[:, [0, 2]], axis=[1, 2]) - start[1]

    assert contours.shape[1:] == (x.shape[0], x.shape[2]), (contours.shape, x.shape)
    return x.astype(np.float32), contours.astype(np.float32)


def remove_background(x):
    # otsu -> greatest connected component -> convex hull
    mask = get_greatest_component(x > threshold_otsu(x))
    mask = apply_along_axes(lambda s: convex_hull_image(s) if s.any() else s, mask, (0, 1))
    return x * mask


def normalize_image(x):
    # using robust normalization that takes into account only values between 10 an 90 percentiles
    return normalize(x, percentiles=10, dtype=np.float32) * (x != 0)


def interpolate_contours(contours, height):
    empty = np.full(height, np.nan)
    y_coords = np.arange(height)

    for contour in contours:
        if not contour:
            yield empty
        else:
            contour = np.array(contour)
            assert contour.ndim == 2 and contour.shape[1] == 2, contour.shape
            ys, xs = contour.T
            assert np.all(np.diff(ys) >= 0)

            yield interp1d(ys, xs, bounds_error=False, fill_value=np.nan)(y_coords)

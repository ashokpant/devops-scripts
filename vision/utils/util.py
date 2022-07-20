import logging
import os
import re
import tarfile
from collections import defaultdict

import numpy as np
import six
import six.moves.urllib as urllib
from six.moves.urllib.parse import urlparse

logger = logging.getLogger(__name__)

img_ext = ['.jpg', '.bmp', '.png', '.JPG', '.JPEG']
vid_ext = ['.mov', '.flv', '.mpeg', '.mpg', '.mp4', '.mvk', '.avi', '.3gp', '.webm']


def maybe_download_ssd_mobilenet(directory='./data'):
    model_file = 'ssd_mobilenet_v1_coco_11_06_2017.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    maybe_download_and_extract(url=download_base + model_file, directory=directory)
    num_classes = 90


def maybe_download_and_extract(url, directory="./"):
    file_path = maybe_download(url=url, directory=directory)
    return untar_file(file_path)


def maybe_download(url, directory="./", filename=None):
    if filename is None:
        filename = os.path.basename(url)
    """Download filename from url unless it's already in directory."""
    if not os.path.exists(directory):
        print("Creating directory %s" % directory)
        os.mkdir(directory)
    file_path = os.path.join(directory, filename)
    if not os.path.exists(file_path):
        print("Downloading %s to %s" % (url, file_path))
        file_path, _ = urllib.request.urlretrieve(url, file_path)
        stat_info = os.stat(file_path)
        print("Successfully downloaded", filename, stat_info.st_size, "bytes")
    return file_path


def untar_file(gz_path, new_path=None):
    """Unzips from gz_path into new_path."""
    if new_path is None:
        new_path = os.path.dirname(gz_path)
    filename = os.path.splitext(os.path.basename(gz_path))[0]
    if gz_path.endswith("tar.gz"):
        new_path = os.path.splitext(new_path)[0]
        filename = os.path.splitext(filename)[0]
        tar = tarfile.open(gz_path, "r:gz")
        tar.extractall(path=new_path)
        tar.close()
    elif gz_path.endswith("tar"):
        tar = tarfile.open(gz_path, "r:")
        tar.extractall(path=new_path)
        tar.close()
    return new_path + "/" + filename


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_files(directory, extensions=None, shuffle=False):
    """
    Lists files in a directory
    :return:
    """

    if extensions is None:
        extensions = img_ext
    images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)

            if extensions is not None:
                if file_path.endswith(tuple(extensions)):
                    images.append(file_path)
            else:
                images.append(file_path)
    if shuffle:
        np.random.shuffle(images)
    return images


def write_files_to_file(directory, filename, shuffle=False):
    files = get_files(directory, shuffle=shuffle)
    file = open(filename, 'w')
    file.writelines('\n'.join(files))


def read_file(filename):
    """
    :param filename:
    :return: list of lines in the given file
    """
    return [line.rstrip('\n') for line in open(filename)]


def list_images(source, shuffle=False):
    """
    Generic function to list images from directory or file
    :return:  list of images in a given directory or file
    """
    files = []
    if os.path.isfile(source):
        if source.endswith('.txt'):
            files = read_file(source)
        elif source.endswith(tuple(img_ext)):
            files.append(source)
        if shuffle:
            np.random.shuffle(files)
    elif os.path.isdir(source):
        files = get_files(source, shuffle=shuffle)

    return files


def list_videos(source, shuffle=False):
    """
    Generic function to list videos from directory or file
    :return:  list of videos in a given directory or file
    """
    files = []
    if os.path.isfile(source):
        if source.endswith('.txt'):
            files = read_file(source)
        elif source.endswith(tuple(vid_ext)):
            files.append(source)

        if shuffle:
            np.random.shuffle(files)
    elif os.path.isdir(source):
        files = get_files(source, extensions=vid_ext, shuffle=shuffle)
    elif isinstance(source, int) or source.isdigit():
        files.append(source)
    return files


def list_images_or_videos(source, shuffle=False):
    """
        Generic function to list images or videos from directory or file
        :return:  list of images in a given directory or file
        """
    files = []
    if os.path.isfile(source):
        if source.endswith('.txt'):
            files = read_file(source)
        elif source.endswith(tuple(img_ext)):
            files.append(source)
        elif source.endswith(tuple(vid_ext)):
            files.append(source)
        if shuffle:
            np.random.shuffle(files)
    elif os.path.isdir(source):
        files = get_files(source, extensions=img_ext + vid_ext, shuffle=shuffle)
    elif isinstance(source, int) or source.isdigit():
        files.append(source)

    return files


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_base_name(path):
    base = os.path.splitext(os.path.basename(path))[0]

    if not base:
        base = path
    base = re.sub('[:/]+', '_', base)
    base = re.sub('^[_+]|[_+]$', '', base)
    return base


def as_text(bytes_or_text, encoding='utf-8'):
    """Returns the given argument as a unicode string.

  Args:
    bytes_or_text: A `bytes`, `str`, or `unicode` object.
    encoding: A string indicating the charset for decoding unicode.

  Returns:
    A `unicode` (Python 2) or `str` (Python 3) object.

  Raises:
    TypeError: If `bytes_or_text` is not a binary or unicode string.
  """
    if isinstance(bytes_or_text, six.text_type):
        return bytes_or_text
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text.decode(encoding)
    else:
        raise TypeError('Expected binary or unicode string, got %r' % bytes_or_text)


def as_str_any(value):
    """
      Args:
    value: A object that can be converted to `str`.

  Returns:
    A `str` object.
  """
    if isinstance(value, bytes):
        return as_text(value)
    else:
        return str(value)


def bounding_boxes(points):
    """
    :param points: [[y, x]]
    :return: [top, left, bottom, right]
    """

    if points is None or len(points) == 0:
        return []

    def _bounding_box(yx):
        y, x = zip(*yx)
        bbox = [min(y), min(x), max(y), max(x)]
        return bbox

    if len(points.shape) == 3:
        return np.array(list(map(lambda points_: _bounding_box(points_), points)))
    else:
        return _bounding_box(points)


def calculate_margin(rect1, rect2):
    """
    Calculate top, left, bottom and right margins between two rectangles
    :param rect2: [top, left, bottom, right]
    :param rect1:  [top, left, bottom, right]
    :return: margin  [top, left, bottom, right]
    """
    top = rect1[0] - rect2[0]
    left = rect1[1] - rect2[1]
    bottom = rect1[2] - rect2[2]
    right = rect1[3] - rect2[3]

    return [top, left, bottom, right]


def calculate_margins(rects1, rects2):
    return np.array(list(map(lambda rect1, rect2: calculate_margin(rect1, rect2), rects1, rects2)))


def area(bbox):
    """
    :param bbox: [top, left, bottom, right, optional[score]]
    :return: area
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def scales(rects1, rects2):
    """
    Calculate scale of rectangle1 by rectangle2
    :param rects1:
    :param rects2:
    :return:
    """

    def _scale(rect1, rect2):
        area1 = area(rect1)
        area2 = area(rect2)
        if area1 == 0 or area2 == 0:
            return 0.0
        return area1 / area2

    if len(rects1.shape) == 2:
        return np.array(list(map(lambda rect1, rect2: _scale(rect1, rect2), rects1, rects2)))
    else:
        return _scale(rects1, rects2)


def is_url(path):
    try:
        result = urlparse(path)
        return result.scheme and result.netloc and result.path
    except:
        return False


def is_image(path):
    try:
        if path.endswith(tuple(img_ext)):
            return True
    except:
        return False


def mkdirs(path):
    os.makedirs(path, exist_ok=True)


def label2id(labels):
    if len(labels) > 1:
        labels = np.sort(np.unique(labels))
    d = defaultdict()
    for i, label in enumerate(labels, start=1):
        d[label] = i
    return d

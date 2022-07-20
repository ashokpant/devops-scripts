"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 1/22/18
-- Time: 4:46 PM
"""
import base64
import logging
import time
from io import StringIO

import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.externals.joblib import Parallel, delayed

from entities import Product

start = time.time()
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
logger = logging.getLogger()


def imread(filename, size=None, rgb=True):
    try:
        image = cv2.imread(filename)
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if size is not None:
            image = imresize(image, width=size[0], height=size[1])
        return image
    except Exception as e:
        logger.warning("Unable to read image: {}, {}".format(filename, e))
        return None


def resize(frame):
    r = 640.0 / frame.shape[1]
    dim = (640, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return frame


def imresize(img, width=None, height=None):
    size = np.shape(img)
    h = size[0]
    w = size[1]
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height))
    elif width is not None:
        wpercent = (width / float(w))
        hsize = int((float(h) * float(wpercent)))
        img = cv2.resize(img, (width, hsize))
    elif height is not None:
        hpercent = (height / float(h))
        wsize = int((float(w) * float(hpercent)))
        img = cv2.resize(img, (wsize, height))
    return img


def resize_mjpeg(frame):
    r = 320.0 / frame.shape[1]
    dim = (320, 200)  # int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return frame


def crop(image, box, dlib_mode=False):
    if not dlib_mode:
        x, y, w, h = box
        return image[y: y + h, x: x + w]

    return image[box.top():box.bottom(), box.left():box.right()]


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_boxes(image, rects, dlib_mode):
    if dlib_mode:
        image = draw_rects_dlib(image, rects)
    else:
        image = draw_rects_cv(image, rects)
    return image


def draw_rects_cv(img, rects, color=(0, 40, 255)):
    overlay = img.copy()
    output = img.copy()
    for x, y, w, h in rects:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def draw_rects_dlib(img, rects, color=(0, 255, 255)):
    overlay = img.copy()
    output = img.copy()

    for bb in rects:
        bl = (bb.left(), bb.bottom())  # (x, y)
        tr = (bb.right(), bb.top())  # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color, thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def rectangle(image, bbox, text=None, color=(0, 255, 255), thickness=None, use_normalized_coordinates=False):
    size = np.shape(image)
    height, width = size[0], size[1]
    if use_normalized_coordinates:
        bbox = denormalize_bbox(width=width, height=height, bbox=bbox)

    thick = thickness
    if thick is None:
        thick = int((height + width) // 600)

    top, left, bottom, right = bbox
    image = cv2.rectangle(image, (left, top), (right, bottom), color, thick)

    if text is not None:
        cv2.putText(img=image, text=text, org=(int(left), int(top) - 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=color, thickness=thick)
    return image


def draw_bounding_box_on_image(image, box, color='red', thickness=4):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)


def draw_object_info_on_image(image, objects, color=(0, 255, 0), thickness=1,
                              use_normalized_coordinates=True):
    def draw_(image, obj):
        text = ''
        if obj.track_id is not None:
            text += str(obj.track_id) + ":"
        text += obj.name
        if obj.id is not None:
            text += " - " + obj.id
        text += " (" + str(obj.score) + ")"
        image = rectangle(image=image, bbox=obj.bbox, text=text, color=color,
                          use_normalized_coordinates=use_normalized_coordinates, thickness=thickness)
        return image

    if not isinstance(objects, Product):
        for obj in objects:
            image = draw_(image=image, obj=obj)
        return image
    else:
        return draw_(image=image, obj=objects)


def draw_text(image, text, point):
    """
    :param image:
    :param text:
    :param point:  (tl, br)
    :return:
    """
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.25, color=(152, 255, 204), thickness=1)


def draw_rect(img, x, y, w, h, color=(0, 40, 255)):
    overlay = img.copy()
    output = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def draw_rects_dlib(img, rects):
    overlay = img.copy()
    output = img.copy()
    for bb in rects:
        bl = (bb.left(), bb.bottom())  # (x, y)
        tr = (bb.right(), bb.top())  # (x+w,y+h)
        cv2.rectangle(overlay, bl, tr, color=(0, 255, 255), thickness=2)
        cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    return output


def pre_processing(image):
    """Performs CLAHE on a greyscale image"""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(grey)
    return cl1


def rgb_pre_processing(image):
    """Performs CLAHE on each RGB components and rebuilds final
    normalised RGB image - side note: improved face detection not recognition"""
    (h, w) = image.shape[:2]
    zeros = np.zeros((h, w), dtype="uint8")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    (B, G, R) = cv2.split(image)
    R = clahe.apply(R)
    G = clahe.apply(G)
    B = clahe.apply(B)

    filtered = cv2.merge([B, G, R])
    cv2.imwrite('not_filtered_rgb.jpg', image)
    cv2.imwrite('filtered_rgb.jpg', filtered)
    return filtered


def detect_people_hog(image):
    image = rgb_pre_processing(image)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(image, winStride=(30, 30), padding=(16, 16), scale=1.1)
    filtered_detections = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            filtered_detections.append(r)
    image = draw_rects_cv(image, filtered_detections)

    return image


def bgr2rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])


def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def decode(img_str):
    return cv2.imdecode(np.fromstring(img_str, dtype=np.uint8), -1)


def encode(img, convert_color=False):
    if convert_color:
        return cv2.imencode('.jpg', rgb2bgr(img))[1].tobytes()
    else:
        return cv2.imencode('.jpg', img)[1].tobytes()


class VideoSaver():
    def __init__(self, filename, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.videoWriter = cv2.VideoWriter(
            'output_{}'.format(filename), fourcc, fps, (width, height))

    def save(self, frame):
        self.videoWriter.write(frame)

    def close(self):
        try:
            self.videoWriter.release()
        except Exception as e:
            print(e)


def preview1(frame_iter, winname="Video", wait_ms=None, cv=True, color_correct=True):
    if cv:
        cv2.namedWindow(winname=winname)
        for img in frame_iter:
            if color_correct:
                img = rgb2bgr(img)
            cv2.imshow(winname, img)
            if wait_ms is not None:
                cv2.waitKey(wait_ms)

            c = cv2.waitKey(30) & 0xff
            if c == 27 or c == 113:
                break
    else:
        plt.figure()
        for img in frame_iter:
            if color_correct:
                img = rgb2bgr(img)

            plt.imshow(img)
            plt.show(block=True)
            plt.title(winname)
            if wait_ms is not None:
                plt.pause(wait_ms / 1000.0)


def get_generator(fn):
    while True:
        yield fn()


def preview(frame_fn, winname="Video", wait_ms=None, cv=True, color_correct=True):
    if cv:
        cv2.namedWindow(winname=winname)
        while True:
            img = frame_fn()
            if img is None:
                continue
            if color_correct:
                img = rgb2bgr(img)
            cv2.imshow(winname, img)
            if wait_ms is not None:
                cv2.waitKey(wait_ms)

            c = cv2.waitKey(30) & 0xff
            if c == 27 or c == 113:
                cv2.destroyWindow(winname)
                break

    else:
        plt.figure()
        while True:
            img = frame_fn()
            if img is None:
                continue

            if color_correct:
                img = rgb2bgr(img)

            plt.imshow(img)
            plt.show(block=True)
            plt.title(winname)
            if wait_ms is not None:
                plt.pause(wait_ms / 1000.0)


def read_file(filename, sep=' '):
    X = []
    Y = []
    f = open(filename)
    for line in f.readlines():
        tokens = line.split(sep)
        X.append(tokens[0])
        Y.append(tokens[1])
    return np.array(X), np.array(Y)


def read_images(image_list, size=None):
    images = Parallel(n_jobs=4, verbose=5)(
        delayed(imread)(f, size) for f in image_list
    )
    return images


def show_image(image, text=None, winname="Output", pause=0):
    image = image.astype(np.uint8).copy()
    if text is not None:
        emmi = np.full_like(image, 255)
        add_text(img=emmi, text=text, text_top=np.int32(image.shape[1] / 2), text_left=np.int32(0),
                 image_scale=1)
        image = np.hstack((image, np.full_like(image, 255), emmi))

    cv2.imshow(winname, image.astype(np.uint8))
    cv2.waitKey(pause)
    c = cv2.waitKey(30) & 0xff
    if c == 27 or c == 113:
        cv2.destroyAllWindows()


def add_text(img, text, text_top, text_left=0, image_scale=1):
    """
    Args:
        img (numpy array of shape (width, height, 3): input image
        text (str): text to add to image
        text_top (int): location of top text to add
        image_scale (float): image resize scale

    Summary:
        Add display text to a frame.

    Returns:
        Next available location of top text (allows for chaining this function)
    """
    cv2.putText(
        img=img,
        text=text,
        org=(text_left, text_top),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.45 * image_scale,
        color=(0, 0, 0))
    return text_top + int(5 * image_scale)


def imshow(image, winname="", wait_ms=None, cv=True, bgr=True):
    if not cv:
        plt.figure()
        plt.imshow(image)
        plt.show(block=True)
        plt.title(winname)
        if wait_ms is not None:
            plt.pause(wait_ms / 1000.0)
    else:
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        if len(image.shape) == 3 and not bgr:
            image = rgb2bgr(image)
        cv2.imshow(winname, image)
        if wait_ms is not None:
            cv2.waitKey(wait_ms)


def pil2cv(image):
    return np.array(image)


def cv2pil(image):
    return Image.fromarray(image)


def denormalize_bbox(width, height, bbox):
    # ymin, xmin, ymax, xmax = bbox
    # (left, right, top, bottom) = (xmin * width, xmax * width,
    #                               ymin * height, ymax * height)
    top, left, bottom, right = bbox

    left = left * width
    top = top * height

    right = right * width
    bottom = bottom * height
    dbbox = [int(top), int(left), int(bottom), int(right)]
    return dbbox


def denormalize_bboxes(image, bboxes):
    """
    :param image:
    :param bboxes: [[top, left ,  bottom, right, optional[score]] or [top, left ,  bottom, right, optional[score]]
    :return:
    """
    temp = bboxes.copy()
    temp.astype(np.float32)
    size = np.shape(image)
    height, width = size[0], size[1]

    if len(temp.shape) == 2:
        if len(temp[0]) == 4:
            temp = np.multiply(temp, [height, width, height, width])
        elif len(temp[0]) == 5:
            temp = np.multiply(temp, [height, width, height, width, 1])
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return temp
    else:
        if len(temp) == 4:
            temp = np.multiply(temp, [height, width, height, width])
        elif len(temp) == 5:
            temp = np.multiply(temp, [height, width, height, width, 1])
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return temp


def normalize_bbox(width, height, bbox):
    top, left, bottom, right = bbox

    left = left / width
    top = top / height

    right = right / width
    bottom = bottom / height
    return [top, left, bottom, right]


def normalize_bboxes(image, bboxes):
    """
    :param image:
    :param bboxes: [[top, left ,  bottom, right, optional[score]] or [top, left ,  bottom, right, optional[score]]
    :return:
    """
    temp = bboxes.copy()
    temp.astype(np.float32)
    size = np.shape(image)
    height, width = size[0], size[1]

    if len(temp.shape) == 2:
        if len(temp[0]) == 4:
            temp = np.divide(temp, [height, width, height, width])
        elif len(temp[0]) == 5:
            temp = np.divide(temp, [height, width, height, width, 1])
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return temp
    else:
        if len(temp) == 4:
            temp = np.divide(temp, [height, width, height, width])
        elif len(temp) == 5:
            temp = np.divide(temp, [height, width, height, width, 1])
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return temp


def denormalize_coords(image, coords):
    """
    :param image:
    :param coords: [N, [top, left]]
    :return:
    """
    temp = coords.copy()
    size = np.shape(image)
    height, width = size[0], size[1]

    if len(temp.shape) == 3:
        temp = list(map(lambda _points: np.multiply(_points, [height, width]), coords))
    else:
        temp = np.multiply(temp, [height, width])
    return temp


def normalize_coords(image, coords):
    """ Used for facial landmark normalization
    :param image:
    :param coords: [N, [top, left]]
    :return:
    """
    temp = np.array(coords.copy())
    size = np.shape(image)
    height, width = size[0], size[1]

    if len(temp.shape) == 3:
        temp = np.array(list(map(lambda _points: np.divide(_points, [height, width]), coords)))
    else:
        temp = np.divide(temp, [height, width])
    return temp


def tlbr2tlwh(bboxes):
    def _tlbr2tlwh(bbox):
        """
    :param bboxes:  [[top, left ,  bottom, right, optional[score]]] or [top, left ,  bottom, right, optional[score]]
    :return: [[top, left ,  width, height, optional[score]]] or [top, left ,  width, height, optional[score]]
        """
        if len(bbox) == 4:
            box = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        elif len(bbox) == 5:
            box = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[4]]
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return box

    if len(np.shape(bboxes)) == 2:
        temp = list()
        for bbox in bboxes:
            t = _tlbr2tlwh(bbox)
            temp.append(t)
        return np.array(temp)
    else:
        return _tlbr2tlwh(bboxes)


def tlwh2tlbr(bboxes):
    """
    :param bboxes: [[top, left ,  width, height, optional[score]]] or [top, left ,  width, height, optional[score]]
    :return:  [[top, left ,  bottom, right, optional[score]]] or [top, left ,  bottom, right, optional[score]]
    """

    def _tlwh2tlbr(bbox):
        if len(bbox) == 4:
            box = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        elif len(bbox) == 5:
            box = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1], bbox[4]]
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")
        return box

    if len(np.shape(bboxes)) == 2:
        temp = list()
        for bbox in bboxes:
            t = _tlwh2tlbr(bbox)
            temp.append(t)
        return np.array(temp)
    else:
        return _tlwh2tlbr(bboxes)


def ltrb2tlbr(bboxes):
    """
    :param bboxes: [[left, top ,  right, bottom, optional[score]]] or [left, top ,  right, bottom, optional[score]]
    :return:  [[top, left ,  bottom, right, optional[score]]] or [top, left ,  bottom, right, optional[score]]
    """
    temp = bboxes.copy()
    if len(temp.shape) == 2 and temp.shape[0] != 0:
        if temp.shape[0] == 0:
            if len(temp) == 4:
                temp = temp[:, [1, 0, 3, 2]]
            elif len(temp) == 5:
                temp = temp[:, [1, 0, 3, 2, 4]]
            else:
                raise ValueError("Input must be of size [N, 4] or [N, 5]")
        else:
            if len(temp[0]) == 4:
                temp = temp[:, [1, 0, 3, 2]]
            elif len(temp[0]) == 5:
                temp = temp[:, [1, 0, 3, 2, 4]]
            else:
                raise ValueError("Input must be of size [N, 4] or [N, 5]")
    else:
        if len(temp) == 4:
            temp = temp[[1, 0, 3, 2]]
        elif len(temp) == 5:
            temp = temp[[1, 0, 3, 2, 4]]
        else:
            raise ValueError("Input must be of size [N, 4] or [N, 5]")

    return temp


def imcrop(image, bbox):
    """
    :param image:
    :param bbox: [top, left, bottom, right]
    :return:
    """
    top, left, bottom, right = bbox
    return image[top:bottom, left:right]


def text2img(text, size):
    img = Image.new('RGB', size, color=(73, 109, 137))

    fnt = ImageFont.truetype('Arial.ttf', 16)
    d = ImageDraw.Draw(img)
    d.text((10, 10), text=text, font=fnt, fill=(255, 255, 0))

    return np.asarray(img, dtype=np.uint8)


def encode_image(image):
    image_buffer = StringIO()
    image.save(image_buffer, format='PNG')
    imgstr = 'data:image/png;base64,{:s}'.format(
        base64.b64encode(image_buffer.getvalue()))
    return imgstr


def extract_boxes(image, big_box=True):
    """
   :param image: masked image or binary image
   :param big_box:
   :return: [N, [left, top, width, height]] urf [N, [x, y, width, height]]
   """
    boxes = []
    image = image.astype(np.uint8)
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    p, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if big_box:
        cnt = []
        for i in range(0, len(contours)):
            for points in contours[i]:
                cnt.append(points)
        cnt = np.array(cnt)
        if len(cnt) > 0:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, w, h])
    else:
        for i in range(0, len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 30 ** 2 and (
                    (w < image.shape[0] and h <= image.shape[1]) or (w <= image.shape[0] and h < image.shape[1])):
                boxes.append([x, y, w, h])
    return boxes


class VideoSaver(object):
    def __init__(self, filename, width, height, fps):
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # *'XVID'
        self.videoWriter = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    def save(self, frame):
        self.videoWriter.write(frame)

    def close(self):
        try:
            self.videoWriter.release()
        except Exception as e:
            print(e)

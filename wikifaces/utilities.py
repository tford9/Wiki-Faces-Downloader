import os
from pathlib import Path
from typing import Dict

import asks
import cv2
import numpy as np
import trio


class Person:
    name = None
    page_url = None

    images = None
    image_links = None

    face_images = None
    face_image_locations = None
    face_image_objects = None

    def __init__(self, name):
        self.name = name
        self.page_url = None

        self.images = dict()
        self.image_links = list()

        self.face_images = dict()
        self.face_image_locations = dict()
        self.face_image_objects = dict()

    def __str__(self):
        return f" Name: {self.name}\n URL: {self.page_url}\n Image_links: {self.image_links}"

    def __repr__(self):
        return self.__str__()

    def set_name(self, name):
        if self.name is None:
            self.name = name

    def add_image_links(self, image_link: str):
        self.image_links.append(image_link)

    def add_image(self, image_filename, image):
        self.images[image_filename] = image

    def add_face_location(self, face_link, image_location):
        self.face_image_locations[face_link] = image_location

    def add_face_images(self, face_link, face_img):
        self.face_images[face_link] = face_img

    def add_face_object(self, face_link, face_obj):
        self.face_image_objects[face_link] = face_obj

    def remove_face_image(self, face_link):
        self.face_image_locations.pop(face_link, None)
        self.face_images.pop(face_link, None)
        self.face_image_objects.pop(face_link, None)

    def set_page_url(self, url):
        if self.page_url is None:
            self.page_url = url


def verify_file(path):
    return os.path.exists(path)


def verify_dir(path):
    """Given a path to a directory, create it if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def crop(image, bbox, margin=50, square=False, dy_margin=False):
    """Crop the image given bounding box.
    Params:
        image: a numpy array
        bbox: a numpy array [left, top, right, bottom]
        margin: <int> margin for cropping face
    Return:
        patch: a numpy array
    """
    h, w = image.shape[:2]
    if dy_margin:
        face_w, face_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] - face_w / 2, bbox[1] - face_h / 2, bbox[2] + face_w / 2, bbox[
            3] + face_h / 2
    else:
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin
    if square:
        bbox[2] = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) + bbox[0]
        bbox[3] = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) + bbox[1]
    bbox = bbox.astype(int)
    bbox[bbox < 0] = 0
    bbox[2] = min(bbox[2], w)
    bbox[3] = min(bbox[3], h)
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


async def fetch_pic(s, url):
    r = await s.get(url)
    if str(r) != "<Response 200 OK>":
        print(str(r), url)
    return r.content


async def save_pic(s, _id, url):
    try:
        content = await fetch_pic(s, url)
        ext = os.path.splitext(url)[1]
        filename = f"/nfs/datasets/WIT/images/{_id}{ext}"
        with open(filename, 'wb') as f:
            f.write(content)

    except Exception as e:
        print("ERROR:", e)


async def main(links):
    domain_name = 'https://upload.wikimedia.org'
    s = asks.sessions.Session(domain_name, connections=4)
    async with trio.open_nursery() as n:
        for _id, url in links:
            n.start_soon(save_pic, s, _id, url)


def bulk_image_download(dir: str, links):
    trio.run(main, links)


def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    remaining_paths = []
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
        else:
            remaining_paths.append(path)
            
    return remaining_paths

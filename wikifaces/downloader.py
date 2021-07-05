import os
import pickle
import urllib
from argparse import ArgumentParser as ArgP
from io import BytesIO
from time import sleep
from typing import Dict, Set
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
# face detction package
from facenet_pytorch import MTCNN
from mediawiki import MediaWiki, MediaWikiPage
from tqdm import tqdm

from utilities import Person, verify_dir, verify_file


def crop(image, bbox, margin=20, square=False, dy_margin=False):
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


def retrieve_images(page_names: Dict, output_location='./data/wiki/', doFace_detection=False):
    skipped_downloads = 0
    failed_downloads = 0
    keep_characters = (' ', '.', '_')

    # get the face detection model
    # device = 'cuda:1'
    mtcnn = MTCNN(image_size=256, margin=20, keep_all=True)

    def face_detect(image: Image, file_path, thredshold=0.8):
        face_locations, probs, landmarks = mtcnn.detect(image, landmarks=True)
        face_list = []

        if face_locations is None or len(face_locations) == 0:
            print("   Fail to find face in %s" % (file_path))
        else:
            # iterate all the faces
            for idx, location in enumerate(face_locations):
                if probs[idx] < thredshold:
                    continue
                left, top, right, bottom = location
                bbox = np.array([left, top, right, bottom])
                face = crop(np.array(image), bbox)
                face = Image.fromarray(face)
                face_list.append(face)
        return face_list

    for key, person in tqdm(page_names.items(), desc='Processing Image Downloads'):
        folder_name = key.lower().replace(' ', '_')
        output_path = f'{output_location}/{folder_name}/'
        verify_dir(output_path)

        for idx, l in enumerate(person.image_links):
            # get filename
            _parsed = urlparse(l)
            _filename = os.path.basename(_parsed.path)
            # filename = filename.split('.')[0]
            _filename = urllib.parse.unquote(_filename)
            _filename = "".join(c for c in _filename if c.isalnum() or c in keep_characters).rstrip()
            # check if file already exists in the location
            if verify_file(output_path + str(_filename)):
                skipped_downloads += 1
                break
            try:
                r = requests.get(l)
                if not doFace_detection:
                    # the output images may not actually be jpgs
                    output_filename = os.path.abspath(f'{output_path}/{_filename}')
                    with open(output_filename, 'wb') as f:
                        f.write(r.content)
                    person.add_image_location(l, output_location)
                else:
                    # convert the binary data into an image
                    try:
                        img = Image.open(BytesIO(r.content)).convert("RGB")
                    except Exception as e:
                        print(f"    Cannot identify image file {l}")

                    faces = face_detect(img, l)
                    if not len(faces):
                        continue
                    ext_idx = _filename.rfind(".")
                    _filename_ = _filename[:ext_idx]
                    _ext_ = _filename[ext_idx:]
                    for idx, face in enumerate(faces):
                        # _filename, _ext = os.path.splitext(_filename)
                        output_filename = os.path.abspath(f'{output_path}/{_filename_}-p{idx}.jpg')
                        face.save(output_filename)
                    person.add_image_location(l, output_location)

            except ConnectionError as e:
                failed_downloads += 1
                continue
    print(f'Skipped Downloads: {skipped_downloads}')
    print(f'Failed Downloads: {failed_downloads}')
    return page_names


if __name__ == '__main__':

    parser = ArgP()
    parser.add_argument('-i', '--categories', dest='categories', nargs='*',
                        help='mediawiki categories to search from (i.e. `indonesian politicians`)')
    parser.add_argument('-o', '--output', dest='output_location',
                        help='Location where output will be saved')
    parser.add_argument('-d', '--face-detection', dest='doFace_detection', action='store_true',
                        help='enable do face detection during the image downloading')
    args = parser.parse_args()

    print(args.output_location)
    print(args.categories)
    initial_categories = set(args.categories)
    first_cat = args.categories[0].replace(' ', '_')
    wikipedia = MediaWiki(rate_limit=False)
    print('Collecting Initial Categories...')
    categories = wikipedia.categorytree(list(initial_categories), depth=1)
    print(f'Initial Categories Collected: {len(categories)}')
    people_pages = {}

    cat_output_directory = args.output_location + '/' + first_cat + '/'
    verify_dir(cat_output_directory)


    def get_data(category, results):
        data = None
        attempts = 0
        failed = True
        while data is None and attempts < 10:
            attempts += 1
            try:
                data = wikipedia.categorymembers(category=category, results=results)
                failed = False
            except Exception as e:
                failed = True
                data = None
                print(f'ReadTimeout')
                sleep(10 * attempts)
        if failed:
            print(f'ReadTimeout caused failure to retrieve after repeated timeouts. For Subcat {category}')
            return set(), category
        return data


    def get_pages(category, seen_categories=None, depth=0, max_depth=2) -> Set:
        if seen_categories is None:
            seen_categories = set()
        pbar.update(1)
        pgs: Set = set()  # pages collected
        scs: Set = set()  # categories collected
        unseen_categories = category.difference(seen_categories)
        for cat in unseen_categories:
            data = get_data(category=cat, results=10000)
            pgs.update(data[0])
            scs.update(data[1])
            seen_categories.add(cat)
        if depth >= max_depth or len(scs) == 0:
            pbar2.update(1)
            return pgs

        pgs.update(get_pages(scs, seen_categories=seen_categories, depth=depth + 1, max_depth=max_depth))
        pbar2.update(1)
        return pgs


    print('Getting Extended Categories...')
    max_depth = 2
    cache_filename = f'{cat_output_directory}cached_{args.categories[0]}.pkl'
    if not verify_file(cache_filename):
        pbar = tqdm(total=max_depth, desc="Descent")
        pbar2 = tqdm(total=max_depth, desc="Ascent")
        pages = get_pages(initial_categories, max_depth=max_depth)
        with open(cache_filename, 'wb') as f:
            pickle.dump(pages, file=f)
    else:
        with open(cache_filename, 'rb') as f:
            pages = pickle.load(f)

    cache_filename = f'{cat_output_directory}/cached_{len(initial_categories)}_dis_pages.pkl'
    dis_pages = []
    if not verify_file(cache_filename):
        print(f'Number of Pages Collected: {len(pages)}')
        nonperson_pages = 0
        for p in tqdm(pages, desc="Processing pages"):
            nonperson_pages += 1
            if not p.startswith('Template:') and not p.startswith('Wikipedia:'):
                # attempt to pull the page and handle disambiguation errors are they appear
                try:
                    page: MediaWikiPage = wikipedia.page(title=p, auto_suggest=False)
                except Exception as e:
                    dis_pages.append(e)
                if any(['people' in cat for cat in page.categories]) or 'born' in page.summary:
                    nonperson_pages -= 1
                    if page.categories and not any([' stubs' in cat for cat in page.categories]):
                        for img_link in page.images:
                            parsed = urlparse(img_link)
                            filename = os.path.basename(parsed.path)
                            if ' ' in p and any([True for x in p.lower().split(' ') if x in filename.lower()]):
                                if p not in people_pages:
                                    people_pages[p] = Person(p)
                                    people_pages[p].set_page_url(page.url)
                                people_pages[p].add_image_links(img_link)
        with open(cache_filename, 'wb') as f:
            pickle.dump(people_pages, file=f)
        print("Pages ignored: ", nonperson_pages)
    else:
        with open(cache_filename, 'rb') as f:
            people_pages = pickle.load(f)
    print(f"Dis Pages: {len(dis_pages)}")

    doFace_detection = args.doFace_detection
    print(f"We are doing face detect: {doFace_detection}")
    retrieve_images(people_pages, output_location=cat_output_directory, doFace_detection=doFace_detection)

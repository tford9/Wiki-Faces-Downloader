import os
import pathlib
import pickle
import urllib
from argparse import ArgumentParser as ArgP
from io import BytesIO
from time import sleep
from typing import Any, Dict, Set, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image
from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from mediawiki import MediaWiki, MediaWikiPage
from tqdm import tqdm

from wikifaces.utilities import Person, verify_dir, verify_file


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


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


class WikiFace:
    user_agent = "Wikifaces-Downloader/1.1 (https://trentonford.com/; tford5@nd.edu) pyMediaWiki/0.70"
    user_agent_dict = {"User-Agent": user_agent}

    wikidata = None
    output_location: Union[str, pathlib.Path] = None
    ascent_pbar: tqdm = None
    descent_pbar: tqdm = None
    mtcnn = None

    face_detection_model = None

    def __init__(self):
        # create wiki object - defaults to Wikipedia.org
        self.wikidata = MediaWiki(rate_limit=False,
                                  user_agent=self.user_agent)

    def face_match(self, person: Person, detect_faces: bool = True, face_match: bool = True,
                   match_threshold: float = 0.5) -> Person:

        mod_person = person
        if not detect_faces:
            return mod_person

        image_to_face_lookup = dict()
        faces_found = 0

        for image_filename, image in mod_person.images.items():
            # if face detection object not loaded
            if self.face_detection_model is None:
                self.face_detection_model = FaceAnalysis()
                self.face_detection_model.prepare(ctx_id=0)

            cv2_img = pil_to_cv2(image)

            image_to_face_lookup[image_filename] = list()

            face_locations, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
            if face_locations is None or len(face_locations) == 0:
                print(f'No faces found in {image_filename}')
            else:
                # iterate all the faces
                if len(face_locations) == 1:
                    for idx, location in enumerate(face_locations):
                        if probs[idx] < 0.95:
                            continue

                        left, top, right, bottom = location
                        bbox = np.array([left, top, right, bottom])
                        face = crop(cv2_img, bbox)
                        mod_person.add_face_images(faces_found, face)
                        output_filename = os.path.abspath(f'{image_filename}-p{faces_found}.jpg')
                        mod_person.add_face_location(faces_found, output_filename)
                        image_to_face_lookup[image_filename].append(faces_found)
                        faces_found += 1

        if not face_match:
            return mod_person
        # image_to_face_lookup = dict(sorted(image_to_face_lookup.items(), key=lambda item: len(item[1])))

        # for img_fn in image_to_face_lookup.keys():
        #     if len(image_to_face_lookup[img_fn]) == 1:

        # if len(mod_person.face_images) > 2:
        #     # faces_object_list: List = self.face_detection_model.get(cv2_img)
        #     face_vectors = {}
        #     for key, face_image in mod_person.face_images.items():
        #         face_object: List = self.face_detection_model.get(face_image)
        #         if face_object:
        #             face_vectors[key] = face_object[0].normed_embedding
        #
        #     X = list()
        #     Y = list()
        #     [(X.append(x), Y.append(y)) for y, x in face_vectors.items()]
        #
        #     db: DBSCAN = DBSCAN(eps=match_threshold, min_samples=1).fit(X)
        #     cluster_sizes = np.bincount(db.labels_)
        #     largest_cluster = (np.argmax(cluster_sizes))
        #
        #     for idx, cluster_id in enumerate(db.labels_):
        #         if cluster_id != largest_cluster:
        #             mod_person.remove_face_image(Y[idx])
        return mod_person

    def retrieve_images(self, page_names: Dict, output_location: Union[str, Any] = './data/wiki/',
                        detect_faces: bool = True, face_consistency_checking: bool = True,
                        face_match_threshold: float = 0.5):
        skipped_downloads = 0
        failed_downloads = 0
        keep_characters = (' ', '.', '_')

        if self.mtcnn is None:
            self.mtcnn = MTCNN(image_size=256, margin=20, keep_all=True)

        for key, person in tqdm(page_names.items(), desc='Processing Image Downloads'):
            folder_name = key.lower().replace(' ', '_')
            output_path = f'{output_location}/{folder_name}/'
            verify_dir(output_path)
            image_extensions = ['.jpg', '.png', '.gif']

            for idx, l in enumerate(person.image_links):
                # get filename
                _parsed = urlparse(l)
                _filename = os.path.basename(_parsed.path)

                # make sure link points to image
                _filename_, _file_extension = os.path.splitext(_filename)
                if _file_extension not in image_extensions:
                    # print(_file_extension)
                    continue

                # filename = filename.split('.')[0]
                _filename = urllib.parse.unquote(_filename)
                _filename = "".join(c for c in _filename if c.isalnum() or c in keep_characters).rstrip()
                # check if file already exists in the location
                if verify_file(output_path + str(_filename)):
                    skipped_downloads += 1
                    break
                try:
                    r = requests.get(l, headers=self.user_agent_dict)
                    output_filename = os.path.abspath(f'{output_path}/{_filename}')

                    if not detect_faces or not face_consistency_checking:
                        # the output images may not actually be jpgs
                        with open(output_filename, 'wb') as f:
                            f.write(r.content)
                        person.add_face_location(l, output_filename)
                    else:
                        # convert the binary data into an image
                        img: Image = None
                        try:
                            img = Image.open(BytesIO(r.content)).convert("RGB")
                        except Exception as e:
                            print(f"Cannot identify image file {l}")
                        person.add_image(output_filename, img)

                except Exception as e:
                    print(f'Image Download Failed:{e}')

            # if best match voting is on, reduce faces list
            person = self.face_match(person=person,
                                     match_threshold=face_match_threshold,
                                     detect_faces=detect_faces,
                                     face_match=face_consistency_checking)

            for key, value in person.face_images.items():
                # _filename, _ext = os.path.splitext(_filename)
                output_filename = person.face_image_locations[key]
                cv2.imwrite(output_filename, value)

        print(f'Skipped Downloads: {skipped_downloads}')
        print(f'Failed Downloads: {failed_downloads}')
        return page_names

    def get_pages(self, category, seen_categories=None, depth=0, max_depth=2) -> Set:
        if seen_categories is None:
            seen_categories = set()
        self.descent_pbar.update(1)
        pgs: Set = set()  # pages collected
        scs: Set = set()  # categories collected
        unseen_categories = category.difference(seen_categories)
        for cat in unseen_categories:
            data = self.get_data(category=cat, results=10000)
            pgs.update(data[0])
            scs.update(data[1])
            seen_categories.add(cat)
        if depth >= max_depth or len(scs) == 0:
            self.ascent_pbar.update(1)
            return pgs

        pgs.update(self.get_pages(scs, seen_categories=seen_categories, depth=depth + 1, max_depth=max_depth))
        self.descent_pbar.update(1)
        return pgs

    def get_data(self, category, results):
        data = None
        attempts = 0
        failed = True
        while data is None and attempts < 10:
            attempts += 1
            try:
                data = self.wikidata.categorymembers(category=category, results=results)
                failed = False
            except Exception as e:
                failed = True
                data = None
                print(f'Backoff Timeout {10 * attempts} Seconds...')
                sleep(10 * attempts)
        if failed:
            print(f'ReadTimeout caused failure to retrieve after repeated timeouts. For Subcategory {category}')
            return set(), category
        return data

    def download(self, categories, output_location='./data/', depth=1, detect_faces=True,
                 face_consistency_checking: bool = True, face_match_threshold=0.5):
        initial_categories = set(categories)
        if len(initial_categories) == 0:
            return 0
        first_cat = categories[0].replace(' ', '_')

        print('Collecting Initial Categories...')
        cat_output_directory = output_location + '/' + first_cat + '/'
        verify_dir(cat_output_directory)

        print('Getting Extended Categories and Page Names...')
        max_depth = depth
        cache_filename = f'{cat_output_directory}cached_pages_d{depth}.pkl'
        if not verify_file(cache_filename):
            self.descent_pbar = tqdm(total=max_depth, desc="Descent")
            self.ascent_pbar = tqdm(total=max_depth, desc="Ascent")
            pages = self.get_pages(initial_categories, max_depth=max_depth)
            with open(cache_filename, 'wb') as f:
                pickle.dump(pages, file=f)
        else:
            with open(cache_filename, 'rb') as f:
                pages = pickle.load(f)

        print(f'Downloading People Pages...')
        cache_filename = f'{cat_output_directory}/cached_{len(initial_categories)}_people_pages_d{depth}.pkl'
        people_pages = {}
        dis_pages = []
        if not verify_file(cache_filename):
            print(f'Number of Pages Collected: {len(pages)}')
            nonperson_pages = 0
            for p in tqdm(pages, desc="Processing pages"):
                nonperson_pages += 1
                if not p.startswith('Template:') and not p.startswith('Wikipedia:'):
                    # attempt to pull the page and handle disambiguation errors are they appear
                    try:
                        page: MediaWikiPage = self.wikidata.page(title=p, auto_suggest=False)
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
            print("None-person pages ignored: ", nonperson_pages)
        else:
            with open(cache_filename, 'rb') as f:
                people_pages = pickle.load(f)
        print(f"People Pages: {len(people_pages)}")

        print(f"Face Detection: {detect_faces}")
        print(f"Face Consistency: {face_consistency_checking}")
        self.retrieve_images(people_pages, output_location=cat_output_directory, detect_faces=detect_faces,
                             face_consistency_checking=face_consistency_checking,
                             face_match_threshold=face_match_threshold)


if __name__ == '__main__':
    parser = ArgP()
    parser.add_argument('-i', '--categories', dest='categories', nargs='*',
                        help='mediawiki categories to search from (i.e. `indonesian politicians`)')
    parser.add_argument('-o', '--output', dest='output_location',
                        help='Location where output will be saved')
    parser.add_argument('-d', '--depth', dest='depth', help='How far from the initial categories to download images.')
    parser.add_argument('-f', '--face-detection', dest='detect_faces', action='store_true',
                        help='enable do face detection during the image downloading')
    args = parser.parse_args()

    wikiface_obj = WikiFace()
    wikiface_obj.download(args.categories, args.output_location, args.doFace_detection, args.depth)

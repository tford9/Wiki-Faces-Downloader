import copy
import os
import pathlib
import pickle
import re
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
from mediawiki import MediaWiki, MediaWikiPage, DisambiguationError
from tqdm import tqdm

from wikifaces.utilities import Person, ResizeWithAspectRatio, crop, pil_to_cv2, remove_empty_folders, verify_dir, \
    verify_file


class WikiFace:
    user_agent = "Wikifaces-Downloader/1.1 (https://trentonford.com/; tford5@nd.edu) pyMediaWiki/0.70"
    user_agent_dict = {"User-Agent": user_agent}

    wikidata = None
    output_location: Union[str, pathlib.Path] = None
    ascent_pbar: tqdm = None
    descent_pbar: tqdm = None
    mtcnn = None
    images_processed = 0
    reinitialize_model_thresh = 100
    count = 0

    face_detection_model = None

    def __init__(self):
        # create wiki object - defaults to Wikipedia.org
        self.wikidata = MediaWiki(rate_limit=False,
                                  user_agent=self.user_agent)
        self.mtcnn = MTCNN(image_size=256, margin=20, keep_all=True)

    def face_match(self, person: Person, detect_faces: bool = True, face_match: bool = True,
                   match_threshold: float = 0.5) -> Person:

        mod_person = person
        if not detect_faces:
            return mod_person
        self.images_processed += 1

        image_to_face_lookup = dict()
        faces_found = 0

        for image_filename, image in mod_person.images.items():

            try:
                cv2_img = pil_to_cv2(image)
                if len(cv2_img) == 0:
                    raise Exception
            except Exception as e:
                print(f"pil_to_cv2 failed on image: {image_filename} with image data:{image}")
                continue

            image_to_face_lookup[image_filename] = list()

            # rescale if necessary
            if cv2_img.shape[:2][0] > 1280 or cv2_img.shape[:2][1] > 1280:
                cv2_img = ResizeWithAspectRatio(cv2_img, width=1280)

            try:
                face_locations, probs, landmarks = self.mtcnn.detect(cv2_img, landmarks=True)
            except Exception as e:
                return mod_person

            if face_locations is None or len(face_locations) != 1:
                # print(f'No faces found in {image_filename}')
                pass
            else:
                face_list = []
                for idx, face in enumerate(face_locations):
                    if probs[idx] > 0.95:
                        face_list.append(idx)

                # iterate all the faces
                for idx in face_list:
                    if len(face_list) == 1:

                        left, top, right, bottom = face_locations[idx]
                        bbox = np.array([left, top, right, bottom])
                        face = crop(cv2_img, bbox, dy_margin=True)
                        if face.size == 0:
                            continue
                        mod_person.add_face_images(faces_found, face)
                        output_filename = os.path.abspath(f'{image_filename}-p{faces_found}.jpg')
                        mod_person.add_face_location(faces_found, output_filename)
                        image_to_face_lookup[image_filename].append(faces_found)
                        faces_found += 1

        # if not face_match:
        #     return mod_person
        return mod_person

    def retrieve_images(self, page_names: Dict, output_location: Union[str, Any] = './data/wiki/',
                        detect_faces: bool = True, face_consistency_checking: bool = True,
                        face_match_threshold: float = 0.5):
        skipped_downloads = 0
        failed_downloads = 0
        keep_characters = (' ', '.', '_')
        simp_idx = 1

        for key, person in tqdm(page_names.items(), desc='Processing Image Downloads'):
            # print(person)
            # continue
            # if simp_idx >= 590:
            #     print('here')
            # print(person)
            # else:
            #     simp_idx +=1
            #     continue

            if self.mtcnn is None or self.count > self.reinitialize_model_thresh:
                self.mtcnn = MTCNN(image_size=256, margin=20, keep_all=True)
                self.count = 0

            self.count += 1
            self.images_processed += 1

            _person = copy.deepcopy(person)
            folder_name = '_'.join(re.findall('[\w\d\_]+', key.lower().replace(' ', '_')))
            output_path = f'{output_location}/{folder_name}/'
            verify_dir(output_path)
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

            for idx, l in enumerate(_person.image_links):
                # get filename
                _parsed = urlparse(l)
                _filename = os.path.basename(_parsed.path)

                # make sure link points to image
                _filename_, _file_extension = os.path.splitext(_filename)
                if _file_extension.lower() not in image_extensions:
                    print(f'Failed to download:{_filename}')
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

                    if not detect_faces and not face_consistency_checking:
                        # the output images may not actually be jpgs
                        with open(output_filename, 'wb') as f:
                            f.write(r.content)
                        _person.add_face_location(l, output_filename)
                    else:
                        # convert the binary data into an image
                        img: Image = None
                        try:
                            img = Image.open(BytesIO(r.content)).convert("RGB")
                        except Exception as e:
                            print(f"Cannot identify image file {l}")
                        _person.add_image(output_filename, img)

                except Exception as e:
                    print(f'Image Download Failed:{e}')

            # if best match voting is on, reduce faces list
            heavy_person = self.face_match(person=_person,
                                           match_threshold=face_match_threshold,
                                           detect_faces=detect_faces,
                                           face_match=face_consistency_checking)

            for key, value in heavy_person.face_images.items():
                # _filename, _ext = os.path.splitext(_filename)
                output_filename = heavy_person.face_image_locations[key]
                try:
                    # print(output_filename)
                    cv2.imwrite(output_filename, value)
                except Exception as e:
                    print(f'imwrite failed on file {output_filename} with key: {key}\n Error {e}')
            del heavy_person
            del _person
            # del person
        print(f'Skipped Downloads: {skipped_downloads}')
        print(f'Failed Downloads: {failed_downloads}')

    def get_pages(self, category, seen_categories=None, depth=0, max_depth=2) -> Set:
        if seen_categories is None:
            seen_categories = set()
        try:
            self.descent_pbar.update(1)
        except Exception as e:
            print('Running in Debug Mode.')
        pgs: Set = set()  # pages collected
        # get immediate pages from disambiguation
        scs: Set = set()  # categories collected
        unseen_categories = category.difference(seen_categories)

        for cat in unseen_categories:
            try:
                pgs.update([self.wikidata.page(title=cat, auto_suggest=False)])
            except DisambiguationError as e:
                pages = self.wikidata.opensearch(cat)
                pages = [page for page in pages if page[0] != cat]
                pgs.update([page[0] for page in pages if ':' not in page[0]])
            except Exception as e:
                pages = self.wikidata.opensearch(cat)
                pgs.update([page[0] for page in pages if ':' not in page[0]])


        for cat in unseen_categories:
            data = self.get_data(category=cat, results=10000)
            pgs.update(data[0])
            scs.update(data[1])
            seen_categories.add(cat)

        if depth >= max_depth or len(scs) == 0:
            try:
                self.ascent_pbar.update(1)
            except Exception as e:
                print('Running in Debug Mode.')
            return pgs

        pgs.update(self.get_pages(scs, seen_categories=seen_categories, depth=depth + 1, max_depth=max_depth))
        try:
            self.descent_pbar.update(1)
        except Exception as e:
            print('Running in Debug Mode.')
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

        people_related_categories = ['']
        people_related_keywords = ['person', 'character', 'actor', 'human' , 'male', 'female', 'man', 'woman']

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
                        continue

                    # search for people-like categories in the page data
                    people_category_found = False
                    for item in page.categories:
                        for item2 in people_related_keywords:
                            if item2 in item:
                                people_category_found = True
                                break

                    if people_category_found or 'born' in page.summary:
                        if page.categories and not any([' stubs' in cat for cat in page.categories]):
                            nonperson_pages -= 1
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

        # clean empty directories from the category output directory
        remove_empty_folders(cat_output_directory)


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
    # wikiface_obj.download(args.categories, args.output_location, args.doFace_detection, args.depth)
    wikiface_obj.download(['indonesia'], depth=6, face_match_threshold=0.5)

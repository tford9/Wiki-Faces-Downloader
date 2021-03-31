import os
from argparse import ArgumentParser as ArgP
from typing import Dict
from urllib.parse import urlparse

import requests
from mediawiki import MediaWiki

from src.utilities import verify_dir, verify_file


class Person:
    name = ''
    image_links = []
    image_locations = []

    def __init__(self, name):
        self.name = name

    def add_image_links(self, image_link):
        self.image_links.append(image_link)


def retrieve_images(page_names: Dict, wikiObj, output_location='/home/tford5/faces_datasets/wikipedia/'):
    skipped_downloads = 0
    for key, person in page_names.items():
        folder_name = key.lower().replace(' ', '_')
        output_path = f'{output_location}/{folder_name}/'
        verify_dir(output_path)

        for idx, link in enumerate(person.image_links):
            # get filename
            parsed = urlparse(link)
            filename = os.path.basename(parsed.path)
            filename = filename.split('.')[0]
            # check if file already exists in the location
            if verify_file(output_path + str(filename)):
                skipped_downloads += 1
                break
            r = requests.get(link)
            # the output images may not actually be jpgs
            output_filename = os.path.abspath(f'{output_path}/{filename}.jpg')
            with open(output_filename, 'wb') as f:
                f.write(r.content)
            person.image_locations.append(output_location)
            break
    print(f'Skipped Downloads: {skipped_downloads}')
    return page_names


if __name__ == '__main__':

    parser = ArgP()
    parser.add_argument('-i', '--categories', dest='categories', nargs='*',
                        help='mediawiki categories to search from (i.e. `indonesian politician`)')
    parser.add_argument('-o', '--output', dest='output_location',
                        help='Location where output will be saved')
    args = parser.parse_args()

    print(args.output_location)
    print(args.categories)
    # exit()
    keys = args.categories
    wikipedia = MediaWiki()
    categories = wikipedia.categorytree(keys, depth=500)
    people_pages = dict()

    for k in keys:
        related_categories = categories[k]['parent-categories']
        for r in related_categories:
            # print(r)
            pages = wikipedia.categorymembers(category=r, results=1000, subcategories=True)[0]
            # print(pages)
            # print(type(pages))
            for p in pages:
                if not p.startswith('Template:'):
                    if not p.startswith('Wikipedia:'):
                        # attempt to pull the page and handle disambiguation errors are they appear
                        page = wikipedia.page(title=p, auto_suggest=False)
                        person = Person(p)
                        # print([' stubs' in cat for cat in page.categories], page.categories)
                        if page.categories and not any([' stubs' in cat for cat in page.categories]):
                            image_links = page.images
                            for link in image_links:
                                parsed = urlparse(link)
                                filename = os.path.basename(parsed.path)
                                print(filename)
                                if ' ' in p:
                                    if any([True for x in p.split(' ') if x in filename]):
                                        # if p.split(' ')[0] in filename or p.split(' ')[1] in filename:
                                        print(page.url)
                                        if p not in people_pages:
                                            people_pages[p] = person
                                        else:
                                            person = people_pages[p]
                                        person.add_image_links(link)
    print(people_pages)
    retrieve_images(people_pages, wikipedia, output_location=args.output_location)

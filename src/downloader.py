import os
from argparse import ArgumentParser as ArgP
from typing import Dict, Set
from urllib.parse import urlparse

import requests
from mediawiki import MediaWiki, MediaWikiPage

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
    categories = wikipedia.categorytree(keys, depth=50)
    people_pages = dict()


    # yes it's recursive
    def get_pages(category, depth=0) -> Set:
        pgs: Set = set()
        scs: Set = set()
        for cat in category:
            data = wikipedia.categorymembers(category=cat, results=1000, subcategories=True)
            pgs.update(data[0])
            scs.update(data[1])
        if depth >= 50 or len(scs) == 0:
            return pgs
        pgs.update(get_pages(scs, depth=depth + 1))
        return pgs

        # for k in keys:
        related_categories = categories[k]['parent-categories']
        # for r in related_categories:
        # print(r)


    pages = get_pages(keys)
    nonperson_pages = 0
    for p in pages:
        nonperson_pages += 1
        if not p.startswith('Template:') and not p.startswith('Wikipedia:'):
            # attempt to pull the page and handle disambiguation errors are they appear
            page: MediaWikiPage = wikipedia.page(title=p, auto_suggest=False)
            if any(['people' in cat for cat in page.categories]) or 'born' in page.summary:
                nonperson_pages -= 1
                person = Person(p)
                if page.categories and not any([' stubs' in cat for cat in page.categories]):
                    image_links = page.images
                    for link in image_links:
                        parsed = urlparse(link)
                        filename = os.path.basename(parsed.path)
                        # print(filename)
                        tokenized_name = []
                        if ' ' in p and any([True for x in p.split(' ') if x in filename]):
                            print(page.url)
                            if p not in people_pages:
                                people_pages[p] = person
                            else:
                                person = people_pages[p]
                        person.add_image_links(link)
    print(people_pages)
    print("Pages ignored: ", nonperson_pages)
    retrieve_images(people_pages, wikipedia, output_location=args.output_location)

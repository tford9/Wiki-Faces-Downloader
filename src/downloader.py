import os
from typing import Dict
from urllib.parse import urlparse

import requests
from mediawiki import MediaWiki, MediaWikiPage

#
from src.utilities import verify_dir


class Person:
    name = ''
    image_links = []
    image_locations = []

    def __init__(self, name):
        self.name = name

    def add_image_links(self, image_link):
        self.image_links.append(image_link)


def retrieve_images(page_names: Dict, wikiObj, output_location='/home/tford5/faces_datasets/wikipedia/'):
    for key, person in page_names.items():
        folder_name = key.lower().replace(' ', '_')
        output_path = f'{output_location}/{folder_name}/'
        verify_dir(output_path)
        for idx, link in enumerate(person.image_links):
            r = requests.get(link)
            parsed = urlparse(link)
            filename = os.path.basename(parsed.path)
            filename = filename.split('.')[0]
            # the output images may not actually be jpgs
            output_filename = os.path.abspath(f'{output_path}/{filename}.jpg')
            with open(output_filename, 'wb') as f:
                f.write(r.content)
            person.image_locations.append(output_location)
            break
    return page_names


if __name__ == '__main__':
    keys = ['indonesian politician']
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
                        page: MediaWikiPage = wikipedia.page(title=p)
                        person = Person(p)
                        # print([' stubs' in cat for cat in page.categories], page.categories)
                        if page.categories and not any([' stubs' in cat for cat in page.categories]):
                            image_links = page.images
                            for link in image_links:
                                parsed = urlparse(link)
                                filename = os.path.basename(parsed.path)
                                # print(filename)
                                # print(p)
                                if ' ' in p:
                                    if p.split(' ')[0] in filename or p.split(' ')[1] in filename:
                                        print(page.url)
                                        if p not in people_pages:
                                            people_pages[p] = person
                                        else:
                                            person = people_pages[p]
                                        person.add_image_links(link)

    print(people_pages)
    retrieve_images(people_pages, wikipedia)

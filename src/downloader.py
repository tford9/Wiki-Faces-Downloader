import os
from argparse import ArgumentParser as ArgP
from typing import Dict, Set
from urllib.parse import urlparse

import requests
from mediawiki import MediaWiki, MediaWikiPage

from src.utilities import Person, verify_dir, verify_file


def retrieve_images(page_names: Dict, wikiObj, output_location='/data/tford5/faces_datasets/wikipedia/'):
    skipped_downloads = 0
    for key, person in page_names.items():
        folder_name = key.lower().replace(' ', '_')
        output_path = f'{output_location}/{folder_name}/'
        verify_dir(output_path)

        for idx, l in enumerate(person.image_links):
            # get filename
            parsed = urlparse(l)
            filename = os.path.basename(parsed.path)
            filename = filename.split('.')[0]
            # check if file already exists in the location
            if verify_file(output_path + str(filename)):
                skipped_downloads += 1
                break
            r = requests.get(l)
            # the output images may not actually be jpgs
            output_filename = os.path.abspath(f'{output_path}/{filename}.jpg')
            with open(output_filename, 'wb') as f:
                f.write(r.content)
            person.add_image_location(l, output_location)
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
    initial_categories = args.categories
    wikipedia = MediaWiki()
    categories = wikipedia.categorytree(initial_categories, depth=50)
    people_pages = {}


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

        # for k in initial_categories:
        # related_categories = categories[k]['parent-categories']
        # for r in related_categories:
        # print(r)


    pages = get_pages(initial_categories)
    nonperson_pages = 0
    for p in pages:
        nonperson_pages += 1
        if not p.startswith('Template:') and not p.startswith('Wikipedia:'):
            # attempt to pull the page and handle disambiguation errors are they appear
            page: MediaWikiPage = wikipedia.page(title=p, auto_suggest=False)
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
    # print(people_pages)
    # for x in people_pages.items():
    #     print(x)
    print("Pages ignored: ", nonperson_pages)
    retrieve_images(people_pages, wikipedia, output_location=args.output_location)

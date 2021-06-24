import os
import pickle
import urllib
from argparse import ArgumentParser as ArgP
from time import sleep
from typing import Dict, Set
from urllib.parse import urlparse

import requests
from mediawiki import MediaWiki, MediaWikiPage
from tqdm import tqdm

from utilities import Person, verify_dir, verify_file


def retrieve_images(page_names: Dict, output_location='./data/wiki/'):
    skipped_downloads = 0
    failed_downloads = 0
    keep_characters = (' ', '.', '_')
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
                # the output images may not actually be jpgs
                output_filename = os.path.abspath(f'{output_path}/{_filename}')
                with open(output_filename, 'wb') as f:
                    f.write(r.content)
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
                        help='mediawiki categories to search from (i.e. `indonesian politician`)')
    parser.add_argument('-o', '--output', dest='output_location',
                        help='Location where output will be saved')
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
            return (set(), category)
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
    max_depth = 25
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
    retrieve_images(people_pages, output_location=cat_output_directory)

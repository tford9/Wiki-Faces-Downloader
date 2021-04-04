import os
import pickle
from argparse import ArgumentParser as ArgP
from typing import Dict, Set
from urllib.parse import urlparse

import requests
from mediawiki import MediaWiki, MediaWikiPage
from tqdm import tqdm

from src.utilities import Person, verify_dir, verify_file


def retrieve_images(page_names: Dict, wikiObj, output_location='/data/tford5/faces_datasets/wikipedia/'):
    skipped_downloads = 0
    for key, person in tqdm(page_names.items(), desc='Processing Image Downloads'):
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
    initial_categories = set(args.categories)
    wikipedia = MediaWiki(rate_limit=False)
    print('Collecting Initial Categories...')
    categories = wikipedia.categorytree(list(initial_categories), depth=1)
    print(f'Initial Categories Collected: {len(categories)}')
    people_pages = {}


    # yes it's recursive
    def get_pages(category, seen_categories=set(), depth=0, max_depth=50) -> Set:
        pbar.update(1)
        pgs: Set = set()  # pages collected
        scs: Set = set()  # categories collected
        unseen_categories = category.difference(seen_categories)
        for cat in unseen_categories:
            data = wikipedia.categorymembers(category=cat, results=100, subcategories=True)
            pgs.update(data[0])
            scs.update(data[1])
            seen_categories.add(cat)
        if depth >= max_depth or len(scs) == 0:
            pbar2.update(1)
            return pgs

        pgs.update(get_pages(scs, seen_categories=seen_categories, depth=depth + 1, max_depth=max_depth))
        pbar2.update(1)
        return pgs

        # for k in initial_categories:
        # related_categories = categories[k]['parent-categories']
        # for r in related_categories:
        # print(r)


    print('Getting Extended Categories...')
    max_depth = 50
    cache_filename = f'/data/tford5/faces_datasets/wikipedia/cached_{len(initial_categories)}.pkl'
    if not verify_file(cache_filename):
        pbar = tqdm(total=max_depth, desc="Descent")
        pbar2 = tqdm(total=max_depth, desc="Ascent")
        pages = get_pages(initial_categories, max_depth=max_depth)
        with open(cache_filename, 'wb') as f:
            pickle.dump(pages, file=f)
    else:
        with open(cache_filename, 'rb') as f:
            pages = pickle.load(f)

    print(f'Number of Pages Collected: {len(pages)}')
    nonperson_pages = 0
    dis_pages = []
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
    # print(people_pages)
    # for x in people_pages.items():
    #     print(x)
    print("Pages ignored: ", nonperson_pages)
    print(f"Dis Pages: {len(dis_pages)}")
    retrieve_images(people_pages, wikipedia, output_location=args.output_location)

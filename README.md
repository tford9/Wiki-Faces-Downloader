# Wiki Faces Downloader

This project downloads images from a Wiki that include human faces. Specifically, images that are associated with
certain wikipedia categories. The process is carried out as follows:

1. Given a category from a Wiki, collect *n* pages that contain the same category as well as at least one category
   containing "people" in the title.
2. With those pages, crawl across their included categories and collect *y* pages that contain those categories as well
   as at least one "people" category.
3. Given the collected Wiki pages, download the primary image from the page and determine if it is a human face using
   light facial detection.
4. We capture all images from the wiki that contain the name of the page (if it's a person then the filename contains
   their name),
5. Using the captured name and images, we create a dataset for that face.

TODOs:

1. Currently, a part of this process uses a recursive call structure to get all related pages; there may be a way to
   linearize, or parallelize this.
2. Face detection could reduce the image footprint by removing spurious image collection.

# Wiki Faces:

[![License: MIT License](https://img.shields.io/badge/License%3A-MIT%20License-red)](https://mit-license.org/)
<figure>
  <img src="https://github.com/tford9/Wiki-Faces-Downloader/blob/main/Joko_Widodo_Wiki.png" style="width:100%">
  <figcaption>Figure 1: Joko Widodo's Wikipedia page, which includes am image of his face. The cropped image on the right is download into a directory named "Joko_Widodo."</figcaption>
</figure>

## TLDR

This project downloads images from a Wiki that include human faces. Specifically, images that are associated with
certain wikipedia categories.

The process is carried out as follows:

1. Given a category from a Wiki, collect *n* pages that contain the same category as well as at least one category
   containing "people" in the title.
2. With those pages, crawl across their included categories and collect *y* pages that contain those categories as well
   as at least one "people" category.
3. Given the collected Wiki pages, download the primary image from the page and determine if it is a human face using
   light facial detection.
4. We capture all images from the wiki that contain the name of the page (if it's a person then the filename contains
   their name),
5. Using the captured name and images, we create a dataset for that face.

## Installation

<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />

Pip Installation Procedure:

#### From TestPyPI:

```
pip install -i https://test.pypi.org/simple/ wikifaces-tford5
```

#### From Repo:

```
git clone git@github.com:tford9/Wiki-Faces-Downloader.git
cd Wiki-Faces-Downloader
python setup.py
pip install wikifaces
```

TODOs:

1. Currently, a part of this process uses a recursive call structure to get all related pages; there may be a way to
   linearize, or parallelize this.
2. Face detection could reduce the image footprint by removing spurious image collection.

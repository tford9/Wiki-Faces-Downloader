import os


class Person:
    # name = None
    # page_url = None
    # image_links = list()
    # image_locations = dict()

    def __init__(self, name):
        self.name = name
        self.image_links = list()
        self.image_locations = dict()
        self.page_url = None

    def __str__(self):
        return f" Name: {self.name}\n URL: {self.page_url}\n Image_links: {self.image_links}"

    def __repr__(self):
        return self.__str__()

    def set_name(self, name):
        if self.name is None:
            self.name = name

    def add_image_links(self, il: str):
        self.image_links.append(il)

    def add_image_location(self, image_link, image_location):
        self.image_locations[image_link] = image_location

    def set_page_url(self, url):
        if self.page_url is None:
            self.page_url = url


def verify_file(path):
    return os.path.exists(path)


def verify_dir(path):
    """Given a path to a directory, create it if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

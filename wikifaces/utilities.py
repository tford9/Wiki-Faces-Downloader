import os


class Person:
    name = None
    page_url = None

    images = None
    image_links = None

    face_images = None
    face_image_locations = None
    face_image_objects = None

    def __init__(self, name):
        self.name = name
        self.page_url = None

        self.images = dict()
        self.image_links = list()

        self.face_images = dict()
        self.face_image_locations = dict()
        self.face_image_objects = dict()

    def __str__(self):
        return f" Name: {self.name}\n URL: {self.page_url}\n Image_links: {self.image_links}"

    def __repr__(self):
        return self.__str__()

    def set_name(self, name):
        if self.name is None:
            self.name = name

    def add_image_links(self, image_link: str):
        self.image_links.append(image_link)

    def add_image(self, image_filename, image):
        self.images[image_filename] = image

    def add_face_location(self, face_link, image_location):
        self.face_image_locations[face_link] = image_location

    def add_face_images(self, face_link, face_img):
        self.face_images[face_link] = face_img

    def add_face_object(self, face_link, face_obj):
        self.face_image_objects[face_link] = face_obj

    def remove_face_image(self, face_link):
        self.face_image_locations.pop(face_link, None)
        self.face_images.pop(face_link, None)
        self.face_image_objects.pop(face_link, None)

    def set_page_url(self, url):
        if self.page_url is None:
            self.page_url = url


def verify_file(path):
    return os.path.exists(path)


def verify_dir(path):
    """Given a path to a directory, create it if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

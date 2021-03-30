import os


def verify_file(path):
    return os.path.exists(path)


def verify_dir(path):
    """Given a path to a directory, create it if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)

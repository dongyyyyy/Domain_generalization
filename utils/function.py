import os
import random

def search(dirname):
    path_list = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext =='.png':
                path_list.append("%s/%s" % (path, filename))
    path_list.sort()
    random.shuffle(path_list)
    return path_list

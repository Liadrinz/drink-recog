from config import conf

def get_label_vector(name):
    return conf('labels').index(name)
    # return [(1 if name == i else 0) for i in conf('labels')]

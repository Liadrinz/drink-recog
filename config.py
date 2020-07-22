import json

with open('config.json', 'rb') as f:
    config = json.loads(f.read().decode('utf-8'))

def conf(exp):
    chain = exp.split('.')
    p = config
    for prop in chain:
        p = p[prop]
    return p
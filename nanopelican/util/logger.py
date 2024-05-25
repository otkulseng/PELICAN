from pathlib import Path
import os
import yaml

def create_directory(name):
    counter = 0
    while True:
        folder = Path.cwd() / f'{name}-{counter}'
        if not folder.exists():
            break
        counter += 1

    os.mkdir(folder)
    return folder

def save_config_file(filename, data):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
import datetime
import colorsys
import json
import os

def create_metadata(model_name: str) -> dict:
    """ not necessary if we use mlflow """
    metadata = {}
    metadata['modelname'] = model_name
    metadata['timestamp'] = datetime.datetime.now().isoformat()
    return metadata

def save_metadata(data, path):
    """ write metadata file into JSON file """
    with open(os.path.join(path, 'metadata.json'), 'w') as f:
        json.dump(data, f)

def create_category_dict(categories):
    """ transform categories into dictionary """
    print('Generating categories dict')
    categories_dict = {}
    for i in range(len(categories) + 1):
        if i == 0:
            categories_dict['background'] = 0
        else:
            categories_dict[categories[i - 1]] = i
    return categories_dict

def HSVToRGB(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(n): 
    huePartition = 1.0 / (n + 1) 
    return list((HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)))

def create_color_map(categories):
    # background color set to black
    color_map = {'background': (0, 0, 0)}
    colors = getDistinctColors(len(categories))
    for category,color in zip(categories,colors):
        color_map[category] = color
    return color_map
from glob import iglob

from pycocotools.coco import COCO


def merge_cocos(coco_collections):
    '''
    Merging COCO collections categories into a single dictionary.
    '''
    merged_coco__cats_dict = {}
    for coco_collection in coco_collections:
        merged_coco__cats_dict.update(coco_collection.cats)
    return merged_coco__cats_dict


def load_cocos(annotations_filenames):
    '''
    Loading COCO collections from a list annotations files.
    '''
    coco_collections = []
    for filename in reversed(list(annotations_filenames)):
        coco_collections.append(COCO(filename))
    print('COCO files loaded')
    return coco_collections


def get_cocos_categories(annotations_dir_path, predictable_categories):
    '''
    Loads COCO collections from a list annotations files and merge them into a single dictionary.
    Also merges the category ids that represents background into a single array.
    '''
    # Getting COCO Dataset
    annotations_filenames = iglob(
        annotations_dir_path + '/**/*.json', recursive=True)

    # Loading coco collections
    coco_collections = load_cocos(annotations_filenames)

    if len(coco_collections) == 0:
        print('No COCO files found')
        return None

    # Merging COCO collections categories
    merged_coco__cats_dict = merge_cocos(coco_collections)

    # Fetching categories and keeping only the one that we want ( aircrack, black_spot )
    CatDict = {}
    for (i, index) in enumerate(merged_coco__cats_dict):
        # If the category is not in the predictable categories, we replace it with background
        if merged_coco__cats_dict[index]['name'] in predictable_categories:
            CatDict[merged_coco__cats_dict[index]['name']] = index
        else:
            if 'background' not in CatDict:
                CatDict['background'] = [0]
            else:
                CatDict['background'].append(index)
    return CatDict

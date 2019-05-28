import argparse
import os
import urllib.request
import numpy as np
from generate_data import generate_dataset


def download(nums=''):
    """
    args: 
    - nums: str, specify how many categories you want to download to your device
    """
    # The file 'categories.txt' includes all categories you want to download as dataset
    with open("./DataUtils/"+nums+"categories.txt", "r") as f:
        classes = f.readlines()
    classes = [c.replace('\n', '').replace(' ', '_') for c in classes]
    print(classes)
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        cls_url = c.replace('_', '%20')
        path = base+cls_url+'.npy'
        print(path)
        urllib.request.urlretrieve(path, './Data/'+c+'.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Quick, Draw! data from Google and then dump the raw data into cache.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--categories', '-c', type=str, default=10, choices=['all', '100', '30', '10'],
                        help='Choose how many categories you want to download to your device.')
    parser.add_argument('--valfold', '-v', type=float,
                        default=0.2, help='Specify the val fold ratio.')
    parser.add_argument('--max_samples_category', '-msc', type=int, default=5000,
                        help='Specify the max samples per category for your generated dataset.')
    parser.add_argument('--download', '-d', type=int,
                        choices=[0, 1], default=0, help='1 for download data, 0 for not.')
    parser.add_argument('--show_random_imgs', '-show', action='store_true',
                        default=False, help='show some random images while generating the dataset.')
    args = parser.parse_args()

    # Download data.
    if args.download == 1:
        download(args.categories)

    # Generate dataset
    generate_dataset(vfold_ratio=args.valfold, max_samples_per_class=args.max_samples_category,
                     show_imgs=args.show_random_imgs)

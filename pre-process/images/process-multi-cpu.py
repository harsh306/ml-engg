"""
Generate preprocesed data in parallel.

author: harsh
date: Nov 2019
"""

from PIL import Image
import os
from multiprocessing import Pool


def downscale(name: str):
    """
    Loads, then processes the image and finally writes them with multiple CPUs
    Here we choose downsample as image operation but it could be anything with the images
    :param name: path to image
    :return: None
    """
    print(name)
    with Image.open(name) as im:
        w, h = im.size
        w_new = int(w / scale)
        h_new = int(h / scale)
        im_new = im.resize((w_new, h_new), Image.ANTIALIAS)

        save_name = os.path.join(output_dir, name.split('/')[-1])
        im_new.save(save_name)


if __name__ == '__main__':
    # Define the input and output image
    input_dir = '/path/to/input/dir'
    output_dir = '/path/to/output/dir'
    scale = 4.

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_list = os.listdir(input_dir)
    image_list = [os.path.join(input_dir, _) for _ in image_list]

    p = Pool()
    """
    pool.map() Here downscale just requires only one argument i.e image path and 
    does not return anything hence pool.map() is used
    
    pool.map_async() if you want to return something from thread and use it further 
    """
    p.map(downscale, image_list)
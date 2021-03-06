import os
import sys
ROOT_DIR = os.path.abspath("./image-super-resolution")
sys.path.append(ROOT_DIR)
import numpy as np
import imageio
from ISR.models import RRDN
from PIL import Image
import argparse

scale = 3
rrdn = RRDN(
    arch_params={'C': 4, 'D': 3, 'G': 64, 'G0': 64, 'T': 10, 'x': scale},
    )

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='')
parser.add_argument('--weights', required=True,
                    default='',
                    metavar="<weights path>",
                    help='path to weights',
                    )
parser.add_argument('--output', required=True,
                    default='',
                    metavar="<output path>",
                    help='path for output images',
                    )
args = parser.parse_args()

if args.weights == '':
    rrdn.model.load_weights('weights/' +
                            'rrdn-C4-D3-G64-G064-T10-x3/' +
                            '2020-12-16_1620/' +
                            'rrdn-C4-D3-G64-G064-T10-x3_best' +
                            '-val_generator_PSNR_Y_epoch100.hdf5')
else:
    rrdn.model.load_weights(args.weights)

if args.output == '':
    output_folder_name = 'submission'
else:
    output_folder_name = args.output

# make submission files
input_folder_name = 'testing/'
if not os.path.exists(output_folder_name):
    os.makedirs(output_folder_name)

# prepare testing file name list
file_names = os.listdir(input_folder_name)
file_names = [file for file in file_names if file.endswith('.png')]
img_list = np.sort(file_names)

# predict images
for img_name in img_list:
    img = Image.open(input_folder_name + img_name)
    lr_img = np.array(img)
    sr_img = rrdn.predict(lr_img)
    Image.fromarray(sr_img)
    output_path = os.path.join(output_folder_name, img_name)
    print('Writing super resolution image {} ...'.format(output_path))
    imageio.imwrite(output_path, sr_img)
print('Super resolution saved successfully.')

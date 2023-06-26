import cv2
import numpy as np
import os
import PIL.Image as Image
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm

Z_START =16
Z_DIMS = 32


OUTPUT_PNG = False
OUTPUT_NPY = True
tile_size = 256
stride = 128
FRAG_IDS = ['1' ,'2a','2b','3']
idxs=range(Z_START, Z_START+Z_DIMS)
print('idxs:',idxs)
INPUT_DIR = Path("data/train")
OUTPUT_DIR = Path(f"crop{tile_size}_{Z_START}_{Z_DIMS}")





def save_tiles(fragment_id):
    images = read_image(fragment_id)
    mask_img = np.array(Image.open(str(INPUT_DIR / f"{fragment_id}" / "mask.png")))
    mask_img = np.pad(mask_img, [(0, tile_size - mask_img.shape[0] % tile_size),
                                 (0, tile_size - mask_img.shape[1] % tile_size)], 'constant')
    inklabels_img = np.array(Image.open(str(INPUT_DIR / f"{fragment_id}" / "inklabels.png")))
    inklabels_img = np.pad(inklabels_img, [(0, tile_size - inklabels_img.shape[0] % tile_size),
                                           (0, tile_size - inklabels_img.shape[1] % tile_size)], 'constant')

    #print('images:',images.shape)
    #print('mask_img:', mask_img.shape)
    #print('inklabels_img:', inklabels_img.shape)



    x1_list = list(range(0, images.shape[1] - tile_size + 1, stride))
    y1_list = list(range(0, images.shape[0] - tile_size + 1, stride))
    count=0
    for y1 in tqdm(y1_list):
        for x1 in x1_list:
            y2 = y1 + tile_size
            x2 = x1 + tile_size
            if mask_img[y1:y2, x1:x2].sum()==0:
                count+=1
                continue
            np.save(str(OUTPUT_DIR / f"{fragment_id}/{fragment_id}_{y1}_{x1}"), images[y1:y2, x1:x2])
            m=inklabels_img[y1:y2, x1:x2]
            m[m>0]=255
            cv2.imwrite(str(OUTPUT_DIR / f"{fragment_id}/{fragment_id}_{y1}_{x1}.png"),m.astype(np.uint8))
    print('skipped=',count)
def read_image(fragment_id):
    images = []

    idxs = range(Z_START, Z_START + Z_DIMS)
    for i in tqdm(idxs):
        #image = cv2.imread(f"data/train/{fragment_id}/surface_volume/{i:02}.tif", 0)
        image = tiff.imread(f"{INPUT_DIR}/{fragment_id}/surface_volume/{i:02}.tif")
        pad0 = (tile_size - image.shape[0] % tile_size)
        pad1 = (tile_size - image.shape[1] % tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    print('!', images.shape)
    return images


for fragment_id in  FRAG_IDS:
    print(f'fragment_id :{fragment_id}')
    os.makedirs(f"{OUTPUT_DIR}/{fragment_id}",exist_ok=True)
    save_tiles(fragment_id)


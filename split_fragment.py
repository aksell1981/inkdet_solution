import os
import cv2
import numpy as np
import tifffile as tiff
from tqdm import tqdm
data_root = 'data/'

fragment_id =2

path_a = 'data/train/2a/'
path_b = 'data/train/2b/'
os.makedirs(path_a+'surface_volume',exist_ok=True)
os.makedirs(path_b+'surface_volume',exist_ok=True)

h = 8300
mask = cv2.imread(data_root + f"train/{fragment_id}/mask.png", 0)
ir = cv2.imread(data_root + f"train/{fragment_id}/ir.png", 0)
ink = cv2.imread(data_root + f"train/{fragment_id}/inklabels.png", 0)
tif_files = os.listdir(data_root + f"train/{fragment_id}/surface_volume/")
tif = tiff.imread(data_root + f"train/{fragment_id}/surface_volume/00.tif")
tifa=tif[0:h,:]
tifb=tif[h:,:]
ma=mask[0:h,:]
mb=mask[h:,:]
ira=ir[0:h,:]
irb=ir[h:,:]
inka=ink[0:h,:]
inkb=ink[h:,:]
cv2.imwrite(path_a +"mask.png", ma.astype(np.uint8))
cv2.imwrite(path_b +"mask.png", mb.astype(np.uint8))
cv2.imwrite(path_a +"ir.png", ira.astype(np.uint8))
cv2.imwrite(path_b +"ir.png", irb.astype(np.uint8))
cv2.imwrite(path_a +"inklabels.png", inka.astype(np.uint8))
cv2.imwrite(path_b +"inklabels.png", inkb.astype(np.uint8))
for f in tqdm(tif_files):
    tif = tiff.imread(data_root + f"train/{fragment_id}/surface_volume/{f}")
    tifa = tif[0:h, :]
    tifb = tif[h:, :]
    cv2.imwrite(path_a + f"surface_volume/{f}",tifa)
    cv2.imwrite(path_b + f"surface_volume/{f}", tifb)








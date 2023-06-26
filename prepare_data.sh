#download 3Dresnet18 and 3sResnet34 weights preitraned on Kinetics-700 dataset
#reference: https://github.com/kenshohara/3D-ResNets-PyTorch
gdown https://drive.google.com/uc?id=1Nb4abvIkkp_ydPFA9sNPT1WakoVKA8Fa
gdown https://drive.google.com/uc?id=1fFN5J2He6eTqMPRl_M9gFtFfpUmhtQc9
#split fragment 2 in two parts (2a,2b)
python split_fragment.py
#make tiles
python make_tiles.py

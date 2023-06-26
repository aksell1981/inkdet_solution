**Vesuvius Challenge - Ink Detection 5th place solution**

This is the training code of the 5th place solution for Kaggle Vesuvius Challenge.

Solution summary : https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417642

Inference code : https://www.kaggle.com/code/aksell7/3dresnet18-3dresnet34-infer/notebook

*HARDWARE:* 

Ubuntu  20.04.5 LTS

CPU AMD® Ryzen 7 3700

32GB RAM

1 x NVIDIA GeForce RTX3090

#SOFTWARE: (python packages are detailed separately in requirements.txt)

Python 3.9.7

CUDA 11.7

cuddn =8.5.0.96

nvidia drivers 515.105.01

1.Place competition data to **data** folder as below:


/data

└── test

└── train

└── sample_submission.csv

*2.Install python packages:*

run **install_dependencies.sh**

3.Prepare data:

run **prepare_data.sh**

*4.Train models:*

run **train.sh**


*Inference:*

https://www.kaggle.com/code/aksell7/3dresnet18-3dresnet34-infer/notebook

Modify models paths if necessary.
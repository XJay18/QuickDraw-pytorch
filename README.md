# QuickDraw-pytorch
This is a demo for CNN models training on Quick, Draw! dataset. Implemented with pytorch :fire:.

# Quick, Draw!
[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset is a collection of 50 million drawings across 345 categories, provided by [googlecreativelab](https://github.com/googlecreativelab). The demo only uses at most 5000 samples from each of the 345 categories. In total, it is trained with 1380000 samples.

# Pytorch Implementation

:point_down: Here are the step-by-step tutorials.

1. Clone the repo to your local device.

    `git clone https://github.com/XJay18/QuickDraw-pytorch.git`

2. Download data from google and generate train&test dataset. You can run this command for example: 

    `python ./DataUtils/prepare_data.py -c 10 -d 1 -show` 

:bulb: hint:
-  `-c` for how many categories you want to download, available choices: `10`, `30`, `100`, `all`. Note that `all` is 345 categories.
-  `-d 1` means that download the data from internet, and `-d 0` means that not download data and just generate train and test dataset from your pre-download data.
- `-show` means that show some random images while generating the dataset.

3. Start training and evaluating for example.

    `python main.py --ngpu 0 -m convnet -e 5`

:key: Please refer to main.py to see the detailed usage. 

# Reference
- [Train a model in tf.keras with Colab, and run it in the browser with TensorFlow.js](https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e)
  
-  [tfjs-converter](https://github.com/tensorflow/tfjs-converter)
  
-  [pytorch2keras](https://github.com/nerox8664/pytorch2keras) (may be used in future since the current demo is not deployed on web using the first Reference) 

# TODO
- [ ] Devise or revise the current model to achieve higher accuracy on [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset).

- [ ] Enlarge the used dataset (i.e, choose more samples from each categories of the dataset).

- [ ] Deploy the demo on web.

# Purpose
:smiley: I started this project with the purpose of improving my ability in coding quickly :rocket:. Also the project will serve as a push on my way to learning more knowledge and experience from others:star:.



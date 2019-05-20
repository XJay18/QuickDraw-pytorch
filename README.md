# QuickDraw-pytorch
This is a demo for CNN models training on quickdraw dataset. Implemented with pytorch.

# Quick, Draw!
[Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset is a collection of 50 million drawings across 345 categories, provided by [googlecreativelab](https://github.com/googlecreativelab). The demo only uses at most 5000 samples from each of the 345 categories. In total, it is trained with 1380000 samples.

# Pytorch Implementation
Using the resnet34 model defined in `torchvision.models` and trained on 2 NVIDIA TITAN XP. 

# Reference
- [Train a model in tf.keras with Colab, and run it in the browser with TensorFlow.js](https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e)
  
-  [tfjs-converter](https://github.com/tensorflow/tfjs-converter)
  
-  [pytorch2keras](https://github.com/nerox8664/pytorch2keras) (may be used in future since the current demo is not deployed on web using the first Reference 

# TODO
- [ ] Devise or revise the current model to achieve higher accuracy on [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset).

- [ ] Enlarge the used dataset (i.e, choose more samples from each categories of the dataset).

- [ ] Deploy the demo on web.

# Purpose
I started this project with the purpose of improving my ability in coding quickly. Also the project will serve as a push on my way to learning more knowledge and experience from others.

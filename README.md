# Towards Icon Design Using Machine Learning

![icon conversion](/assets/conversion.png)

[[video]](https://youtu.be/rFnMdFkjpAE) [[medium post]](https://medium.com/@moses.soh/towards-automatic-icon-design-using-machine-learning-423cbe6710fe) [[poster]](#poster)

## Introduction

I created a model that learns how to turn outlines into stylish and colorful icons. Icon and logo design is difficult — expert human designers choose from line weights, colors, textures and shapes to create beautiful icons such as [these (I'm a big fan)](https://dribbble.com/yoga). But there seems to be a pattern to how each designer make their choices. So, I decided it would be interesting to try to train a model that learns a designer's style, and then takes any freely available icon outlines (e.g. from [the Noun Project](https://thenounproject.com/)), and color and style them exactly as how a designer would have completely automatically.

The icon generator is a convolutional neural network called a U-Net that was trained on an icon set from [Smashicons](smashicons.com). I optimized the generator against the L1 loss and an adversarial loss under a Conditional Generative Adversarial Network (cGAN) setup.

## <a id="poster"></a>Overview of approach

![poster](/assets/poster.svg)

The poster above summarizes the technical approach of tis project. The image below showcases the performance of our best model on test set images from the Smashicon dataset.

![test](/assets/test.png)

## How to use

### Clone the repo

```
git clone https://github.com/mosessoh/iconcolor
```

### Download pre-trained models

```
cd iconcolor
cd models
python fetch_models.py
```

### Inference

```
python color_icon.py assets/demo.png
```

The `color_icon.py` file contains a script to load the pre-trained generator contained in `model/outline2yellow_generator_gan.pth` and use it to colorize an input icon. This is the generator trained against L1 and adversarial loss. If you're getting funky colorizations (the adversarial loss encourages the use of more vibrant colors), the weights for the L1-optimized generator are at `model/outline2yellow_generator.pth`. Note that the model expects a 1 x 1 x 128 x 128 input, and saves the output at `assets/output.png`. If your setup is correct, you should get the following (the outline icon is from [IconBros](https://www.iconbros.com/) and the colored icon is produced by our model):

![before and after](/assets/before_after.gif)

### Training

The `train_model.py` file contains the training script used to train the discriminator using `BCEWithLogitsLoss` and the generator against L1 and adversarial loss. The `model/outline2yellow_discriminator.pth` and `model/outline2yellow_generator_gan.pth` files contain a useful checkpoints so your discriminator and generator do not need to start from scratch.

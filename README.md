# Monet-Style GAN

## Introduction
This repository contains code for a Generative Adversarial Network (GAN) designed to generate Monet-style paintings from photographs. This GAN consists of a generator model and a discriminator model that work together to create artistic images in the style of the famous painter Claude Monet.

![Sample Monet-style Image](sample.png)

## Prerequisites
Before running the code in this repository, make sure you have the following prerequisites installed:
- Python (>=3.6)
- TensorFlow (>=2.0)
- TensorFlow Addons (tfa)
- Matplotlib
- Numpy

## Dataset

To use this GAN, you need to organize your dataset into two directories:

1. `monet_tfrec/`: This directory should contain TFRecord files with Monet paintings.
2. `photo_tfrec/`: This directory should contain TFRecord files with photographs.

## Models

- **Generator**: Transforms input photographs into Monet-style images.
- **Discriminator**: Distinguishes between real Monet paintings and generated Monet-style images.

## Training

The GAN is trained using adversarial loss, cycle consistency loss, and identity loss:

- Adversarial loss: The generator aims to produce Monet-style images that the discriminator cannot distinguish from real Monet paintings.
- Cycle consistency loss: Ensures that the transformation from a photograph to Monet-style and back to a photograph is close to the original.
- Identity loss: Prevents significant changes in Monet paintings when transformed.

## Usage

1. Clone the repository to your local machine.
2. Organize your dataset as described above.
3. Modify the code, dataset paths, and training parameters in `train.py` as needed.
4. Run the training script to train the GAN.
5. Monitor training progress and loss values.
6. Visualize and assess results after training.
7. Use the trained generator to convert photographs into Monet-style images.

## License

This project is open-source and is licensed under the MIT License. You are free to use, modify, and distribute the code as per the terms of the license. See the [LICENSE](LICENSE) file for more details.

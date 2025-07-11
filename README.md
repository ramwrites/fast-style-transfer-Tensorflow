# Fast Style Transfer - Tensorflow
This repository contains my implementation of the **Fast Neural Style Transfer** model using Tensorflow, 
as proposed in the seminal paper *Perceptual Losses for Real-Time Style Transfer and Super-Resolution* by [Justin Johnson et al.](https://arxiv.org/pdf/1603.08155)

While the original architecture uses Batch Normalization, I replaced it with Instance Normalization following the insights from the paper 
*Instance Noralization: The Missing Ingredient for Fast Stylization* by [Dmitry Ulyanov et al.](https://arxiv.org/pdf/1607.08022)

I trained the model on MS-COCO 2014 train dataset. I resized each of the 82k training images to 256x256 and trained on them with a batch size of 4 for 40k iterations(~2 epochs).

I converted the images dataset into TFRecord format (not mandatory) for storage efficiency.
* **Original Dataset Size:** ~13GB
* **TFRecord Size:** ~3GB

Here's the results that my model generated:

<img width="515" height="341" alt="output 2" src="https://github.com/user-attachments/assets/8f327240-0eaa-4e09-a98b-5155e11e9556" />
<img width="515" height="341" alt="output 2" src="https://github.com/user-attachments/assets/edbbc188-b9da-4612-be83-393d997cc71a" />
<img width="515" height="341" alt="output 2" src="https://github.com/user-attachments/assets/a63eb10f-0894-4e2b-b1b4-6333bc2f11df" />

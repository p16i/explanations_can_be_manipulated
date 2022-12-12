# Explanations can be manipulated and Geometry is to blame (unofficial extended code)
Explanation methods aim to make neural networks more trustworthy and interpretable. In this paper, we demonstrate a property of explanation methods which is disconcerting for both of these purposes. Namely, we show that explanations can be manipulated \emph{arbitrarily} by applying visually hardly perceptible perturbations to the input that keep the network's output approximately constant. We establish theoretically that this phenomenon can be related to certain geometrical properties of neural networks. This allows us to derive an upper bound on the susceptibility of explanations to manipulations. Based on this result, we propose effective mechanisms to enhance the robustness of explanations.

---

**Remarks**: This repository extends the original respository with the following changes (See [[Patch]][diff]):
1. LRP-Gamma for VGG16 with the gamma values {0.5, 0.25, 0.1, 0} used in the 1st and 2nd blocks, 3rd block, 4th block, and the 5th and the classification head respectively.
3. Objective function takes into account the preservation of total relevance;
4. The attack script takes a CSV containing of original and target images and over them. It can be run by 
     ```
     $ head n02097474.csv
     original,target
     image1.jpeg,image2.jpeg
     ...
     
     $python run_attack.py --data_dir ~/datasets/imagenet \
          --cuda 
          --seed_file n02097474.csv
     ```

[diff]: https://github.com/pankessel/explanations_can_be_manipulated/compare/master...p16i:explanations_can_be_manipulated:master




<div align="center">
     <img width="50%" src="https://user-images.githubusercontent.com/1214890/206981186-457c4c48-d388-4f16-8492-588da24f44ad.png"/>
     <div>Result from the extended code;</div>
</div>


We also provide the results of all 50 validation images from Class `Tibetan_terrier` (`n02097474`, 200) at https://tubcloud.tu-berlin.de/s/TmZR8Yje3RcXRif.

---
### What we do

We manipulate images so their explanation resembles an _arbitrary_ target map. Below you can see our algorithm in action:

![](gifs/image_r.gif)
![](gifs/expl_r.gif)

In our paper we show how to achieve such manipulations. We discuss their nature and derive an upper bound on how much the explanation can change. Based on this bound we propose &beta;-smoothing, a method that can be applied to any of the considered explanation methods to increase robustness against manipulations.

### &beta;-smoothing
We have demonstrated that one can drastically change the explanation map while keeping the output of the neural network constant.
We argue that this vulnerability can be related to the large curvature of the output manifold of the neural network. We focus on the gradient method.
The fact that the gradient can be drastically changed by slightly perturbing the input along the hypersurface suggests that the curvature of the hypersurface is large.
If we replace the ReLU activations with softplus activations with parameter &beta;, and reduce &beta; we can reduce the curvature of the lines of equal network output. Below you can see the smoothing in action for a two layer neural network.

![](gifs/equipot_r.gif)

### Links

[NeurIPS paper](https://papers.nips.cc/paper/9511-explanations-can-be-manipulated-and-geometry-is-to-blame)

[archiv version](https://arxiv.org/abs/1906.07983)

[google drive](https://drive.google.com/open?id=1TZeWngoevHRuIw6gb5CZDIRrc7EWf5yb)

## Code

### Install

Install dependencies using
     
     pip install -r requirements.txt 

### Usage

Manipulate an image to reproduce a given target explanation using
    
    python run_attack.py --cuda
    
For explanations beyond lrp you need to enable beta_growth so the second derivative of the activations is not zero.

    python run_attack.py --cuda --method gradient --beta_growth

Plot softplus expanations for various values of beta using

    python plot_expl.py --cuda 
    
To download patterns for pattern attribution, please use the following link:

https://drive.google.com/open?id=1RdvAiUZgfhSE8sVF2JOyURpnk1HQ_hZk

Copy the downloaded file in the models subdirectory. 

### License

This repository is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.

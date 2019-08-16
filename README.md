# Adaptive-diverse-capsule-network

This implement is an improved version of real-valued capsule network from our paper《Cv-CapsNet:complex-valued capsule network》 url:https://ieeexplore.ieee.org/document/8744220. In this implement, a multi-scale feature fusion mechanism based on attention is proposed, named adaptive-diverse-model, which eliminates the manual setting of capsule size in coding stage,  we also use the bottleneck from the MobilenetV3 to improve it(adaptive-diverse-model+). results of ablation research can be seen in Results.

## Usage

**Step 1. Clone this repository to local.**
```
git clone https://github.com/Johnnan002/Adaptive-diverse-capsule-network
cd Adaptive-diverse-capsule-network

```
**Step 2. Train the Adaptive-diverse-capsule-network model on CIFAR10**  

Training with default settings:
```
$ python Adaptive-diverse-capsule-network.py

```
More detailed usage run for help:
```
$ python Adaptive-diverse-capsule-network.py -h
```

**Step 3. Test a pre-trained Adaptive-diverse-capsule-network model**

Suppose you have trained a model using the above command, then the trained model will be
saved to `result/trained_model.h5`. Now just launch the following command to get test results.
```
$ python Adaptive-diverse-capsule-network.py -t -w result/trained_model.h5
```
It will output the testing accuracy .
The testing data is same as the validation data. It will be easy to test on new data, 
just change the code as you want 

## Results

    Validation accuracy > 88.5% after 25 epochs.
    About 600 seconds per epoch on a single tesla k80 GPU card
     ________________________________________________________________________________________
    |        Models         |     Parameters      |  Accuracy(25epoch) |       upgrade       |
    |———————————————————————|—————————————————————|————————————————————|—————————————————————|
    |     original model    |       7.99M         |       71.56%       |        —— ——        |
    |———————————————————————|—————————————————————|————————————————————|—————————————————————|
    |     diverse-model     |       5.3M          |       86.7%        |      ↑ 15.14%       |
    |———————————————————————|—————————————————————|————————————————————|—————————————————————|
    | adaptive-diverse-model|       5.3M          |       87.8%        |      ↑ 16.24%       |
    |———————————————————————|—————————————————————|————————————————————|—————————————————————|
    |adaptive-diverse-model+|       5.3M          |       88.5%        |      ↑ 16.94%       |   
    |_______________________|_____________________|____________________|_____________________|
    
      
If you use the code in your research or wish to refer to the baseline results published in the Model , please use the following BibTeX entry. 
```
@ARTICLE{8744220, 
author={X. {Cheng} and J. {He} and J. {He} and H. {Xu}}, 
journal={IEEE Access}, 
title={Cv-CapsNet: Complex-Valued Capsule Network}, 
year={2019}, 
volume={7}, 
number={}, 
pages={85492-85499},  
doi={10.1109/ACCESS.2019.2924548}, 
ISSN={2169-3536}, 
month={},}
    
    

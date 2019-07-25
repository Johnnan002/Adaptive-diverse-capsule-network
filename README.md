# Adaptive-diverse-capsule-network

This implement is an improved version of real-valued capsule network from our paper《Cv-CapsNet:complex-valued capsule network》, We introduce an attentional mechanism for fusing of three levels of features by weights so as to eliminate the manual setting of capsule dimensions in the coding stage. we also use the bottleneck from the MobilenetV3 to improve it.

## Usage

**Step 1. Clone this repository to local.**
```
git clone https://github.com/Johnnan002/Adaptive-diverse-capsule-network.git 
cd Adaptive-diverse-capsule-network

```
**Step 2. Train a CapsNetAdaptive-diverse-capsule-network model on CIFAR10**  

Training with default settings:
```
    python Adaptive-diverse-capsule-network.py

```
More detailed usage run for help:
```
python Adaptive-diverse-capsule-network.py -h
```

**Step 4. Test a pre-trained Adaptive-diverse-capsule-network model**

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
    
    
    
    
    

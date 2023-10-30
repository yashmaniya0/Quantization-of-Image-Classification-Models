# **Quantization of CNN Models**

## **Introduction**
- Optimization of Convolutional Neural Networks using Quantization is an important technique to reduce the computational and memory requirements of CNN models. 
- Quantization involves reducing the precision of the weights and activations of a CNN model, which results in smaller model size, reduced memory footprint, and faster inference time. 
- The goal of optimization using quantization is to strike a balance between model size, accuracy, and computational efficiency. 
- In this way, quantization can make it possible to deploy deep learning models on devices with limited computational resources, while maintaining acceptable performance.

## **Quantization**
The fundamental idea behind quantization is that if we convert the weights and inputs into integer types, we consume less memory and on certain hardware, the calculations are faster. \
However, there is a trade-off: with quantization, we can lose significant accuracy. We will dive into this later, but first let's see why quantization works.
There are major two types of quantization techniques :
1. Post-training Quantization
2. Quantization-aware training

### **1. Post-training Quantization**
- This technique involves quantizing the weights and activations of the model from floating-point numbers to lower-precision fixed-point or integer numbers.
- This technique can be applied to any pre-trained model without retraining or fine-tuning. Its main advantage that it is simple to apply.
- Post-training quantization can reduce the model size by up to 4x and can also speed up the inference time on various hardware platforms, including CPUs, GPUs, and specialized accelerators.
- Lead to a slight degradation in model accuracy due to the loss of precision during quantization. Not suitable for models that require dynamic computations.

### **2. Quantization-Aware Training**
- Quantize the weights during training. 
- First, the model is trained with full-precision weights and activations, and then fine-tuned with quantization-aware training. 
- Here, even the gradients are calculated for the quantized weights. 
- QAT can significantly improve the accuracy of the quantized models and reduce the accuracy loss compared to post-training quantization.

## **Our Contributions**
We have used the Fashion MNIST dataset.
- Tested accuracy of different models on test data :
    1. Normal CNN (with convolution, pooling, dense layers)
    2. MobileNet
    3. VGG16

- Tested accuracy on Float as well as Quantized models. Techniques used :
    1. Post-Training
    2. QAT


## **Results and Conclusions**

| Model      | No. of Parameters | Non-Quantized Model Accuracy | 8-bit QAT Model Accuracy | Non-Quantized Model Size | 8-bit QAT Model Size |
| ---------- | ----------------- | ---------------------------- | ------------------------ | ---------------------------- | ------------------------ |
| Normal CNN | 20,426            | 88.37 %                      | 87.30 %                  | 0.9174 Mb                    | 0.233 Mb                 |
| MobileNet  | 3, 251, 259       | 85.72 %                      | 83.11 %                  | 12.71 Mb                     | 3.36 Mb                  |
| VGG16      | 14, 781, 642      | 82.20 %                      | 71.82 %                  | 57.63 Mb                     | 15.37 Mb                 |

- Quantization results in a significant reduction in model size, with up to 73.33% reduction in model size observed in the case of the VGG16 model.
- In most cases, the accuracy of the 8-bit QAT (Quantization Aware Training) model is slightly lower than that of the non-quantized model. However, the drop in accuracy is relatively small and may be acceptable in resource-constrained environments.
- The effectiveness of quantization may depend on the specific architecture of the CNN model being used.


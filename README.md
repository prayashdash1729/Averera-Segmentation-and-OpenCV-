<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#1-road-segmentation-using-unet-architecture">Road Segmentation Using UNET</a>
      <ol>
        <li><a href="#workflow">Workflow</a></li>
        <li><a href="#predictions-on-test-data">Predictions on Test Data</a></li>
        <li><a href="#predictions-on-iit-bhu-campus-roads">Predictions on IIT BHU Camous Roads</a></li>
      </ol>
    </li>
    <li>
      <a href="#2-semantic-segmentation-of-indian-roads-using-deeplabv3-architecture">Road Segmentation using DeepLabV3+</a>
    </li>
  </ol>
</details>

</br>

---
---


# 1. Road Segmentation Using UNet Architecture

Used the UNet architecture for semantic segmentation on roads.

## Workflow
1. **Image Preprocessing**: Involves reading the image, resizing it, and normalizing the pixel values (i.e. [0-255] -> [0-1]).
2. **Label Preprocessing**: The dataset was for multi-class segmentation, converted it to binary.
3. **Defining the Model**: Compiled the model correctly.
4. **Training the Model**: Saved the weights after training.
5. **Making Predictions**: Defined a preprocessing pipeline for newly fed images.

## Predictions on Test Data
![image](https://user-images.githubusercontent.com/118126264/217278424-d3b9836f-e249-4410-a048-f23b2ed61e09.png)

## Predictions on IIT BHU Campus Roads
![image](https://user-images.githubusercontent.com/118126264/217278568-c0fb9ee7-1a9b-44fb-8422-2db126ab7ba1.png)

</br>

---
---

</br>

# 2. Semantic Segmentation of Indian Roads Using `DeepLabV3+` Architecture

### 1. Used the `IDD Dataset` (by IIIT Hyderabad).
### 2. Created a training pipeline using the `Data API` of TensorFlow.
   - First, loaded all the image and mask addresses.
   - Created a function for reading, resizing, and normalizing the images.
   - Created a `tf.data.Dataset` object and split it into train and validation datasets to load the images and masks in batches.

### 3. Defined the Model Architecture: Used Xception pretrained model as the base model.
### 4. Training Process: Implemented a checkpointer to save only the best weights.
### 5. Predictions: Made predictions on campus roads of IIT BHU for inference and deployed the model in the autonomous vehicle of Team AVERERA.

![image](https://github.com/mitanshu17/DeepLabV3_Segmentation/assets/118126264/165c88c4-7819-42be-9606-5c12d12cc13a)

![image](https://github.com/mitanshu17/DeepLabV3_Segmentation/assets/118126264/dc2cc264-fc72-4b67-be6c-c21f5c62da41)


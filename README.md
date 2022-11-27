# Car-Make-and-Model-Classifier-using-VGG24
The development of a computer vision application that can recognize a certain vehicle model from an image is an intriguing and difficult subject to solve. The difficulty with this issue is that different car models can often look remarkably similar, and the same vehicle can sometimes look different and difficult to identify depending on lighting, angle, and many other variables. To create a model that can recognize a specific vehicle model for this project, I choose to train a convolutional neural network (CNN) known as VGG24 using Fast ai and PyTorch which is a novel approach to existing architecture VGG16.

VGG 24 Architecture:


![VGG24BD](https://user-images.githubusercontent.com/94075388/204139892-78bcc7fd-d66e-4e11-a63f-3309b85f2ba6.png)



# Dataset

For the Dataset: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

There are 16,185 photos of 196 different kinds of cars in the Cars collection. The data has been divided into 8,144 training photos and 8,041 testing images, roughly splitting each class 50-50. Classes are usually given in the Make, Model, and Years categories.

Visualising the Dataset:

![VGG16-DATA](https://user-images.githubusercontent.com/94075388/204139969-7d1fa8d1-6e88-4da7-82db-ccaff6535029.png)


# Result

Accuracy obtained: 53%

With more training and augmentation of data acuuracy will increase 

![ Vgg24_cars_training](https://user-images.githubusercontent.com/94075388/204140803-291271da-69ad-4cd5-97b0-5561752f2f73.png)

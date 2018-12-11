# A Neural Approach to Artistic Style Transfer with Color Preservation

## Introduction and Goal:

With the rapid growth of technology, a schism between the artistic and scientific worlds has begun to form. Within visual art, there is a beautiful, intimate relationship between style and content. Using computer vision and deep learning techniques, we attempt to extract this style and apply it to other images, thereby fusing the creativity of visual art with the algorithmic world of computer vision. In the artistic world, painters try to explore the dynamic relationship between the subject and the artist’s individual style. Therefore, quantitatively separating the style and the content of a given image could allow us merge different styles and subjects from various artists and image sources respectively to create unique images.

The goal of this project is bipartite. Primarily, it is to transform a baseline image (content image C) to fit the artistic style of another image (style image S). We propose using a well trained Convolutional Neural Network (CNN) to capture context and style representations of the input and painting image respectively. Secondarily, we want to pre- serve the color scheme of the original content image by using statistical approaches to map color.

Overall, the goal of this project is to use a CNN, a gradient-based optimization algorithm, and computer vision tools to design a model that can effectively transfer artistic style an texture from a style image to that of a content image, while simultaneously preserving the color distribution of content image in the output.


## Dependancies and Requirements

  - Python v3.6
  - Tensorflow v1.12.0
  - Numpy
  - Matplotlib
  - Pillow
  - IPython
  - OpenCV-Python

In order to download these dependencies, you could either use the Python package manager ```pip3``` for each isolated dependancy, or you could run the following command upon cloning the repository to your system.

```sh
$ pip3 install -r requirements.txt
```

This bash command will recursively download all the dependencies listed above using the ```pip3``` package.


## Running the Code: 

In order to run the code, you have two options:

  - Source Code
  - iPython Notebooks

We would recommend running the iPython notebooks as one can visualize the progress of the algorithm as well as comprehend the various chunks of the algorithm required for color transfer and artistic style transfer.

If you are running the iPython notebook, we would suggest working on a GPU backed instance (like Colab). Or you could manually create your own instance and configure CUDA 9.0 to work with tensorflow.

  - Step 1: Set up GCP instance as follows to use jupyter notebooks (https://medium.com/datadriveninvestor/complete-step-by-step-guide-of-keras-transfer-learning-with-gpu-on-google-cloud-platform-ed21e33e0b1d). The code can be run in a non-GPU environment, but output will take nearly 1-2hrs to be produced.

  - Step 2: Depending on the preferred color transfer scheme, choose the proper Jupyter Notebook in the notebooks directory of this repository, upload it to the GCP instance, and run all cells. You can watch the image form as the epochs continue. The output will be stored in the ./content/output/ directory of your file system. Additionally, you can view the initial color transfered style image in './content/pictures/transfer.jpeg'

  - Step 3: To change the artwork involved, ensure it is in the proper directory (```content``` and ```style```) and proceed to change the source directory paths in the code. To change the image involved, ensure it is in the proper directory (```content``` or ```style```) and proceed to change the source directory paths in the code.

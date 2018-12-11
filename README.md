# A Neural Approach to Artistic Style Transfer with Color Preservation

Introduction and Goal:

With the rapid growth of technology, a schism between the artistic and scientific worlds has begun to form. Within visual art, there is a beautiful, intimate relationship between style and content. Using computer vision and deep learn- ing techniques, we attempt to extract this style and apply it to other images, thereby fusing the creativity of visual art with the algorithmic world of computer vision. In the artis- tic world, painters try to explore the dynamic relationship between the subject and the artist’s individual style. There- fore, quantitatively separating the style and the content of a given image could allow us merge different styles and sub- jects from various artists and image sources respectively to create unique images.

The goal of this project is bipartite. Primarily, it is to transform a baseline image (content image C) to fit the artis- tic style of another image (style image S). We propose us- ing a well trained Convolutional Neural Network (CNN) to capture context and style representations of the input and painting image respectively. Secondarily, we want to pre- serve the color scheme of the original content image by us- ing statistical approaches to map color.

Overall, the goal of this project is to use a CNN, a gradient-based optimization algorithm, and computer vision tools to design a model that can effectively transfer artistic style an texture from a style image to that of a content im- age, while simultaneously preserving the color distribution of content image in the output.


Running the Code: 
Step 1: Set up GCP instance as follows to use jupyter notebooks (https://medium.com/datadriveninvestor/complete-step-by-step-guide-of-keras-transfer-learning-with-gpu-on-google-cloud-platform-ed21e33e0b1d). The code can be run in a non-GPU environment, but output will take nearly 1-2hrs to be produced.

Step 2: Depending on the preferred color transfer scheme, choose the proper Jupyter Notebook in the notebooks directory of this repository, upload it to the GCP instance, and run all cells. You can watch the image form as the epochs continue. The output will be stored in the ./content/output/ directory of your file system. Additionally, you can view the initial color transfered style image in './content/pictures/transfer.jpeg'

Step 3: To change the artwork involved, ensure it is in the proper directory (./content/artwork/) and proceed to change the source directory paths in the code. To change the image involved, ensure it is in the proper directory (./content/pictures/) and proceed to change the source directory paths in the code.
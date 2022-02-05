# DL4J_practice
DL4J practice, simple MNIST classificaiton.

- source from https://github.com/zq2599/blog_demos/tree/master/dlfj-tutorials (cuda 9.2, under old GPU 950M)

- dependency refer from https://community.konduit.ai/t/dl4j-cuda-11-2-running-out-of-memory-on-evaluation-on-ubuntu-20-04/1377/17

- MNIST data set could retrive from https://raw.githubusercontent.com/zq2599/blog_download_files/master/files/mnist_png.tar.gz

#### I modified the pom.xml dependency to fit CUDA 11.2 & cudnn  cuDNN v8.1.1 
and all run well under my Ampere architecture GPU 3060.

### Driver detail

![image](https://user-images.githubusercontent.com/28682192/152645408-84528822-72e8-44cf-998c-e7e790646df4.png)


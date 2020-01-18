# E4040-2019fFALL-PROJECT-RNDM-ms5898-yd2505-fw2322
This course project is a version of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Network](http://arxiv.org/pdf/1312.6082.pdf) use Tensorflow

## Team Member:
Mingfei Sun;  Yifeng Deng;  Fan Wu

## Requirements:
* Python 3.6.9
* Tensorflow 1.1.3
* imageio 2.6.1
* h5py
* Numpy 1.17


## Function and structure of files:
```
e4040-2019fall-project-rndm-ms5898-yd2505-fw2322
|-- modelbest
|   |-- SVHNmodel_1575510762.index
|   |-- SVHNmodel_1575510762.meta
|   |-- checkpoint
|
|-- utils
|   |-- dataprocess.py        //support functions for Data_Preprocess.ipynb 
|   |-- models
|      |-- model11.py         //model of 11 Layers 
|      |-- L9F2.py            //model of 9 Layers with 2 full connected layers
|      |-- L11SP.py           //model of 11 layers but less parameters
|      |-- L11GP.py           //model of 11 layers but greater parameters
|      |-- model8.py          //model of 8 layers for comparison
|      |-- model7.py          //model of 7 layers for comparison
|      |-- model6.py          //model of 6 layers for comparison
|      |-- model5.py          //model of 5 layers for comparison
|      |-- model4.py          //model of 4 layers for comparison
|      |-- model3.py          //model of 3 layers for comparison
|       
|-- Data_Preprocess.ipynb     //Do data preprocess to SVHN dataset and save to mat file
|-- SVHNClassifier.ipynb      //Read Data, train network, get accuracy
|-- Compare_9&11.ipynb        //compare the performance of 9 layer network with 11 layers
|-- E4040.2019Fall.RNDM.report.ms5898.yd2505.fw2322.pdf  //Project report
|-- README
```
## Install:
1. Download or Clone the code
```
git clone https://github.com/cu-zk-courses-org/e4040-2019fall-project-rndm-ms5898-yd2505-fw2322.git
```
2. Download the data set [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)

## Usage:
1. Run Data_Preprocess.ipynb
```
This notebook read the data from SVHN dataset, then do 
the preprocess to the data and save them as .mat format 
which will use later¶
```
2. Run SVHNClassifier.ipynb
```
This notebook do the following things:
* Read data grom .mat files saved in the data preprcesss
* train the network and save the best model
* load the best model and run on the test set
```
3. Run Compare_9&11.ipynb(optional）
```
This is used to compare the performance of 9 layers and laters
network, for some other layers network the method is the same.
```
## Result:
1. Neural network structure
![Graph](https://raw.githubusercontent.com/cu-zk-courses-org/e4040-2019fall-project-rndm-ms5898-yd2505-fw2322/master/img/network_structure.png?token=ANIW3EV22DKFS26S4TGLPYS57Q46E)
```
As shown above, we built the 11-layer network consisting of eight convolution hidden layers
and two fully connected hidden layers (plus one output layer) with the size parameters described 
in the original paper which is said to be tested achieving the best performance. The convolution 
layers’ sizes are 48, 64, 128, 160, 192, 192, 192, 192 with kernel size of 5*5. The max pooling 
kernel size is 2*2 for all and takes stride as 2 every 2 pool layers (like 2, 1, 2, 1,…). 
In our implementation, the best result we got is 84% test accuracy under this model.
```
2. Tensorboard result:
![Graph](https://raw.githubusercontent.com/cu-zk-courses-org/e4040-2019fall-project-rndm-ms5898-yd2505-fw2322/master/img/tensorboard.png?token=ANIW3EXXBISWEOPF4KAITM257Q5UO)

3. Best accuracy on Test set:

![Graph](https://raw.githubusercontent.com/cu-zk-courses-org/e4040-2019fall-project-rndm-ms5898-yd2505-fw2322/master/img/acc.png?token=ANIW3ESUWFAMIPSSPPEFKOS57Q7FM)

## Reference:
[1] Goodfellow, I.J., Bulatov, Y., Ibarz, J., Arnoud, S. and Shet, V., 2013. Multi-digit number recognition from street view imagery using deep convolutional neural networks. arXiv preprint arXiv:1312.6082.





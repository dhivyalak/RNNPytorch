# RNNPytorch
[Deep Learning](https://developer.ibm.com/learningpaths/get-started-with-deep-learning/an-introduction-to-deep-learning/) is vast field that employs artifical neural networks to process data and train a machine learning model. Two approaches used are *[supervised and unsupervised learning](https://developer.ibm.com/learningpaths/get-started-with-deep-learning/deep-learning-architectures/)*. In this tutorial, we will be focusing on autoencoders. An autoencoder is a type of unsupervised learning technique, which means the machine is using and analyzing unlabeled data sets. Using this method, the model is able to learn patterns in the data and learn how to reconstruct the inputs as its outputs after significanlty downsizing it.  

Autoencoders have four main layers: encoder, bottleneck, decoder, and the reconstruction loss.

* The encoder is the given input with reduced dimentionality.
* The bottleneck is the compressed representation of the encoded data.
* The decoder is the reconstructed version of the original output.
* The reconstruction loss is the difference between the original output and the reconstructed output.

>Input -> Encoder -> Bottleneck -> Decoder -> Ouput

The goal of the model is to minimize the difference between the original output and the reconstructed output, or in other words, to reduce the reconstruction loss.

Some popular applciations of autoencoders are image denoising, dimensionality reduction, and feature extraction, to name a few. The following tutorial will touch on some of these applications and introduce basic autoencoder concepts using Tensorflow. Tensorflow is a Python library that is used to create Deep Learning Models. IBM Watson® Studio is a data science platform that provides all of the tools necessary to develop a data-centric solution on the cloud.

In this tutorial, an MNIST data set is used, and it is deployed on IBM Watson Studio on IBM Cloud Pak® for Data.

# Learning Objectives

In the tutorial, you import a Jupyter Notebook that is written in Python into IBM Watson Studio on IBM Cloud Pak for Data as a Service, then run through the Notebook. The Notebook creates an Autoencoder model using Tensorflow based on an MNIST data set, encoding and decoding the data. After running the Notebook, you should understand how TensorFlow builds and executes an Autoencoder. You'll learn how to:

- Run a Jupyter Notebook using Watson Studio on IBM Cloud Pak for Data as a Service
- Build an Autoencoder model using Tensorflow
- Train the model and evaluate the model by performing validation and testing

# Prerequisites

The following prerequisites are required to follow the tutorial:

- An IBM Cloud account
- IBM Cloud Pak for Data

# Estimated time
It should take you approximately **x** hours complete the tutorial.

# Steps 

1. Set up IBM Cloud Pak for Data as a Service.
2. Create a new project and import the Notebook.
3. Read through the Notebook.
4. Run the Notebook.

# Set up IBM Cloud Pak for Data as a Service.

1.Open a browser, and log in to IBM Cloud with your IBM Cloud credentials.

![log into ibm cloud](images/log-into-ibm-cloud.png)

2.Type Watson Studio in the search bar at the top. If you already have an instance of Watson Studio, it should be visible. If so, click it. If not, click Watson Studio under Catalog Results to create a new service instance.

![select watson studio service](images/select-watson-studio-service.jpeg)

3.Select the type of plan to create if you are creating a new service instance. A Lite (free) plan should suffice for this tutorial. Click Create.

![studio lite plan](images/watson-studio-lite-plan.png)

4.Click Get Started on the landing page for the service instance.

![get started on watson studio](images/get-started-watson-studio.png)

This takes you to the landing page for IBM Cloud Pak for Data as a Service.

5.Click your avatar in the upper right, then click Profile and settings under your name.

![CPFD profile and settings](images/cpdaas-profile-and-settings.jpeg)

6.Switch to the Services tab. You should see the Watson Studio service instance listed under your Your Cloud Pak for Data services.

You can also associate other services such as Watson Knowledge Catalog and Watson Machine Learning with your IBM Cloud Pak for Data as a Service account. These are listed under Try our available services.

In the example shown here, a Watson Knowledge Catalog service instance exists in the IBM Cloud account, so it's automatically associated with the IBM Cloud Pak for Data as a Service account. To add any other service (Watson Machine Learning in this example), click Add within the tile for the service under Try our available services.

![associated services](images/cpdaas-associated-services.png)

7. Select the type of plan to create (a Lite plan should suffice), and click Create.
![lite plan](images/machine-learning-lite-plan.png)

After the service instance is created, you are returned to the IBM Cloud Pak for Data as a Service instance. You should see that the service is now associated with Your IBM Cloud Pak for Data as a Service account.

![associated services](images/cpdaas-all-services-associated.png)

# Create a new project and import the Notebook

1. Navigate to the menu (☰) on the left, and choose View all projects. After the screen loads, click New + or New project + to create a new project.

![new project](images/cpdaas-new-project.png)

2.Select Create an empty project.

![empty project](images/cpdaas-empty-project.png)

3.Provide a name for the project. You must associate an IBM Cloud Object Storage instance with your project. If you already have an IBM Cloud Object Storage service instance in your IBM Cloud account, it should automatically be populated here. Otherwise, click Add.

![set project name](images/cpdaas-project-name.png)

4.Select the type of plan to create (a Lite plan should suffice for this tutorial), and click Create.

![select lite plan](images/cos-lite-plan.png)

5.Click Refresh on the project creation page.

![refresh creation page](images/cpdaas-refresh-cos.png)

6.Click Create after you see the IBM Cloud Object Storage instance that you created displayed under Storage.

![create project](images/cpdaas-create-project.png)

7.After the project is created, you can add the Notebook to the project. Click Add to project +, and select Notebook.

![add notebook to proj](images/cpdaas-add-notebook-to-project.png)

8.Switch to the From URL tab. Provide the name of the Notebook as "AutoencoderUsingTensorFlow". Notebook URL as: "https://raw.githubusercontent.com/IBM/dl-learning-path-assets/main/unsupervised-deeplearning/notebooks/AutoEncoders.ipynb"

9.Under the Select runtime drop-down menu, select Default Python 3.7 S (4 vCPU 16 GB RAM). Click Create.

![load the notebook](images/loadingNotebook.png)

10.After the Jupyter Notebook is loaded and the kernel is ready, you can start running the cells in the Notebook.

![image of the notebook](images/notebookImage.png)

    Important: Make sure that you stop the kernel of your notebooks when you are done to conserve memory resources.

![stop the kernel](images/stopKernel.png)

    Note: The Jupyter Notebook included in the project has been cleared of output. If you would like to see the Notebook that has already been completed with output, refer to the [example Notebook](https://raw.githubusercontent.com/IBM/dl-learning-path-assets/main/unsupervised-deeplearning/examples/autoencoders.ipynb).

# Read through the Notebook

Spend some time looking through the sections of the Notebook to get an overview. A Notebook is composed of text (markdown or heading) cells and code cells. The markdown cells provide comments on what the code is designed to do.

You run cells individually by highlighting each cell, then either clicking Run at the top of the Notebook or using the keyboard shortcut to run the cell (Shift + Enter, but this can vary based on the platform). While the cell is running, an asterisk ([*]) appears to the left of the cell. When that cell has finished running, a sequential number appears (for example, [17]).

![code cells before running](images/codeCell.png)

    Note: Some of the comments in the Notebook are directions for you to modify specific sections of the code. Perform any changes as indicated before running the cell.

The Notebook is divided into multiple sections:

1. Introduction
2. Feature Extraction and Dimensionality Reduction
3. Autoencoder Structure
4. Performance
5. Training: Loss Function
6. Code

Section 6 contains the code to create, validate, test, and run the autoencoder model.

# Run the Notebook

1. Run the code cells in the Notebook starting with the ones in section 4. The first few cells bring in the required modules such as TensorFlow, Numpy, reader, and the data set.

    Note: The second code cell checks for the version of TensorFlow. The Notebook works only with TensorFlow version 2.2.0-rc0. Therefore, if an error is thrown here, you need to ensure that you have installed TensorFlow version 2.2.0-rc0 in the first code cell.

![python imports](images/importsInCodeCells.png)

    Note: If you have installed TensorFlow version 2.2.0-rc0 and still get the error, your changes are not being picked up and you must restart the kernel by clicking Kernel->Restart and Clear Output. Wait until all of the outputs disappear, and then your changes should be picked up.

![restart kernel and clear cell output](images/restartAndClearOutput.png)

2. The training, validation, and testing of the model does not happen until the last code cell. Running all 20 epochs will take some time, about 30 minutes at most.

![train 20 epoch](images/trainingModelFor20Epochs.png)

# Summary

In this tutorial, you learned about Autoencoders and ran an implementation on Jupyter Notebook using Watson Studio on IBM Cloud Pak for Data as a Service. Utilizing python libraries such as: Tensorflow, MatPlotLib, and NumPy were also key to create the model.

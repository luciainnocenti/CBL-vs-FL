<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/imgs/FLCBL_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="/imgs/FLCBL_light.png">
  <img alt="Description of image" src="/docs/img/FLCBL_light.png">
</picture>

# CBL vs FL: implementation with Fed-BioMed
This repository extends the FLamby benchmark to the consensus-based learning (CBL) paradigm, presented by Innocenti et al. (2024).
As explained in the related [paper](https://www.biorxiv.org/content/10.1101/2024.05.27.596048v1.abstract), CBL is an alternative to federated learning for collaborative learning:
Given a test point, CBL combines predictions obtained from the different models trained independently by each client on local data. Therefore, CBL is based on an offline routine, where information is exchanged only during inference. 

Experiments are performed using the framework for federated learning [Fed-BioMed](https://fedbiomed.org/).
To run the experiments, Fed-BioMed must first be installed. A tutorial for installation is available [here](https://fedbiomed.org/latest/tutorials/installation/0-basic-software-installation/). 
N.B.: The results presented in the paper are obtained with Fed-BioMed version 4.5.0

## The data

### FeTS
This dataset was published in the context of the [FeTS Challenge](https://fets-ai.github.io/Challenge/). FeTS is a multi-modal dataset, using 4 different brain MRI modalities to provide a multi-class segmentation among the different regions of gliomas. The data is partitioned based on the 23 data acquisition sites. Clients have variable sizes: the smallest has only 4 data points, and the largest has 511. 

We used a pre-trained SegResNet and evaluated it by the average DSC among different classes.

The [data](https://fets-ai.github.io/Challenge/data/) can be downloaded following the instructions on the challenge website, using the **natural partitioning by institution** file to split the data among clients.

To enable faithful reproduction of experiments, the **fets/splits** folder of this repository contains CSV files for use with Fed-BioMed for defining train and test datasets. 
N.B.: When you will define the nodes in Fed-BioMed, you can use these CSV files.

### FedProstate

We obtained this dataset by gathering data from 3 major publicly available datasets on prostate cancer imaging analysis: [Medical Segmentation Decathlon](http://medicaldecathlon.com/), [Promise12](https://zenodo.org/records/8014041), [ProstateX](https://prostatex.grand-challenge.org/), and by a clinical dataset from the Guy St. Thomas Hospital, London, UK (not available for public use yet).  For ProstateX, the segmentation masks are provided in [this GitHub repository](https://github.com/rcuocolo/PROSTATEx_masks).
The FedProstate dataset contains T2-magnetic resonance imaging (MRI) and segmentation masks of the whole prostate. We used the acquisition protocol (MRI with or without endorectal coil) and the scanner manufacturer as splitting criteria. In this way, we obtained a dataset with 6 clients: 4 of them we used for the training, and 2 we kept as independent test sets.

To enable faithful reproduction of experiments, the **fedprostate/splits** folder of this repository contains CSV files for use with Fed-BioMed for defining train and test datasets. 
N.B.: When you will define the nodes in Fed-BioMed, you can use these CSV files.

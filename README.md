# DeepWhistle
This repository is the implementation of our IJCNN 2020 paper:
[Learning Deep Models from Synthetic Data for Extracting Dolphin Whistle Contours](https://arxiv.org/abs/2005.08894)

### <a name="dependency"></a> Dependency
* Ubuntu ≥ 14.04
* Python ≥ 3.6.8
* CUDA ≥ 8.0
* Pytorch ≥ 0.2.0

Other python libraries:
> ```bash
> pip install -r requirements.txt
> ```

### <a name="processed_data"></a> Our processed data
To reproduce our experiment results, you may use our [processed data](https://drive.google.com/open?id=1CDTupjz_nxEfSe1DUL7twn-K09QWvrI3).
Those data are in hdf5 or lmdb format, which can be directly used by our model training code.

|              | Train Data                                 | Test Data     |
|--------------|--------------------------------------------|---------------|
| DWC-I        | DWC-I/train.h5                             | test.h5 |
| DWC-II       | DWC-II/train_postive, train_negative/       | test.h5        |
| DWC-II-Canny | DWC-II-Canny/train_postive, train_negative/ | test.h5        |
| DWC-III      | DWC-III/train_postive, train_negative/      | test.h5        |
| DWC-IV       | DWC-IV/train_postive, train_negative/       | test.h5        |

You may extract the compressed file to under this directory, and move silbido data to 4.Graph_search/silbido_lipu. 
> ```bash
> unzip data.zip
> mv data/silbido/* 4.Graph_search/silbido_lipu/
> ```

You may see the description below to process your own data. 



### <a name="data"></a> Data
The raw data can be downloaded from [mobysound.org](https://www.mobysound.org/workshops_p2.html).
We used [common dolphin](https://www.mobysound.org/workshops/5th_DCL_data_common.zip) and 
[bottlenose dolphin](https://www.mobysound.org/workshops/5th_DCL_data_bottlenose.zip) 
data in 5th workshop for our experiments. The raw data contains audio recording files(.wav) and the 
annotation file(.bin). 

### <a name="data_preprocessing"></a> Data Preprocessing
We did three things:
1. Transfer audio files to spectrogram images. 
2. Transfer ground truth files to binary black/white images with the same size of corresponding spectrogram.
3. Crop images into patches and stored in HDF5 files.

To transfer raw data into images, you may run the code with:
> ```bash
> python 1.Spectrogram_and_GT/generate_traindata.py --audio_dir PATH_TO_AUDIO_FILES  \ 
>   --annotation_dir PATH_TO_ANNOTATION_FILES --output_dir PATH_TO_OUTPUT_SPECTROGRAM
> ```

To generate training HDF5 files, you may configure the paths within 
matlab code file 2.Network_Training_data/generate_train_hdf5.m, and run the matlab file.

### <a name="model_training"></a> Model training
1. Train DWC-I model
> ```bash
> python 3.Network_train_and_test/train.py --data_type h5 \ 
>   --train_data 3.Network_train_and_test/train_DWC-I.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --exp_name DWC-I
> ```
2. Train DWC-II model
> ```bash
> python 3.Network_train_and_test/train.py --data_type lmdb \ 
>   --train_data 3.Network_train_and_test/train_DWC-II.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --exp_name DWC-II
> ```
3. Train DWC-III model
> ```bash
> python 3.Network_train_and_test/train.py --data_type lmdb \ 
>   --train_data 3.Network_train_and_test/train_DWC-III.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --exp_name DWC-III
> ```
4. Train DWC-IV model
> ```bash
> python 3.Network_train_and_test/train.py --data_type lmdb \ 
>   --train_data 3.Network_train_and_test/train_DWC-IV.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --recall_guided True \
>   --recall_val_data 3.Network_train_and_test/val.txt  \
>   --pretrained_model 3.Network_train_and_test/models/DWC-III.pth \
>   --exp_name DWC-IV
> ```

### <a name="model_training"></a> Model inference
To generate confidence map for each spectrogram, you may run:
> ```bash
> python 3.Network_train_and_test/test.py \ 
>   --model_file PATH_TO_YOUR_MODEL \
>   --test_img_dir PATH_TO_YOUR_TEST_IMAGES \
>   --output_dir PATH_TO_YOUR_OUTPUT
> ```
For example, you can run the following code to get confidence maps from DWC-I on our testing dataset. 
> ```bash
> python 3.Network_train_and_test/test.py \ 
>   --model_file 3.Network_train_and_test/models/DWC-I.pth \
>   --test_img_dir data/test_imgs \
>   --output_dir 3.Network_train_and_test/test_results/DWC-I
> ```

### <a name="model_evaluation"></a> Model Evaluation
To extract whistles from confidence map, and evaluate the performance. You may use our modified silbido, and 
run the following code in MATLAB:
> ```bash
> silbido_init
> test_detection_score_batch
> ```

Please find the original silbido package [here](https://roch.sdsu.edu/index.php/software/). 

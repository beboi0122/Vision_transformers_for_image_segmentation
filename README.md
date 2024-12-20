# Vision_transformers_for_image_segmentation

#### Team name

Q épület

#### Members

- Jurásek Jónás PH4QFK
- Pánczél Bence GZI387

#### Project Description

The project contains a University task, that is to use Vision Transformers for image segmentation.
For the project, we use a dataset of road layouts from kaggle.

#### Files in repository

- project

  - requirements.txt: list of python packages needed for the project
  - main.ipynb: jupiter notebook that downloads, analyses and prepares the data
  - baseline_modell.ipynb: jupiter notebook for the training and evaluation of the baseline models
  - vision_transformers_model.ipynb: jupiter notebook for the training and evaluation of the vision transformer model
  - frontend.ipynb: jupiter notebook for gradio frontend
  - PUT_KAGGLE_JSON_HERE: put your own kaggle.json in this folder
  - baseline_models: python files of baseline models
    - DeepLabV3Model.py
    - UNET2D.py
  - models: folder for saving best performing models
  - metadata: saved train, test, validate datasets for reproduction purposes
    - testdata.csv
    - traindata.csv
    - valdata.csv
  - wrapper_modules

    - RoadSegmentationModule.py: lightning module for training models

  - RoadDataLoader.py: dataloader helper class
  - RoadDataset: dataloader helper class
  - loss_and_eval_functions.py: loss and evaluation fuctions

- Documentation.pdf: Documentation
- Dockerfile: For containerization
- README.md: project description

#### Related works

- https://www.kaggle.com/datasets/payne18/road-detection-dataset-with-masks
- https://github.com/BME-SmartLab-VITMMA19/vision-assignment/blob/main/DL_Practice_4_Vision.ipynb

#### Run instructions

Prerequisites of running:

- docker
- kaggle registration
- wandb registration

Before running the docker image, place your kaggle.json file in the project folder.

To build the docker image run the following command:

```
docker build -t vision-transformers .
```

To run the docker image run the following command:

```
docker run --gpus all -m 4g --shm-size=2g -u root -p 8888:8888 -it vision-transformers
```

##### Milestone 1

After running the docker image, to download and can open the jupiter notebook on http://127.0.0.1:8888/notebooks/main.ipynb and run the contents.

##### Milestone 2

After running the docker image, to train and evaluate the baseline model, you can open the jupiter notebook on http://127.0.0.1:8888/notebooks/baseline_modell.ipynb and run the contents.

The training takes a long time, so a pre-trained model is included in the docker image also, which can be more easily evaluated on.

If you want to only evaluate the pre-trained model, run the Imports, Download, Lightning module, Baseline Model, Load Data, and Testing models sections of the notebook.

If you want to train the models and then evaluate it, then run all cells in the notebook. One cell will ask for your wandb token, provide it in the box to continue the training part.

##### Milestone 3

After running the docker image, to train and evaluate the vision transformer model, you can open the jupiter notebook on http://127.0.0.1:8888/notebooks/vision_transformer_model.ipynb and run the contents. This notebook also includes the comparisonb of all models.

A gradio frontend is also included via a seperate notebook. To start the frontend, you can open the jupiter notebook on http://127.0.0.1:8888/notebooks/frontend.ipynb and run the contents. After the notebook has finished, you can reach the frontend on the link given in the notebook output.

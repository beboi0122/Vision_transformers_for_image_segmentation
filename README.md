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
  - PUT_KAGGLE_JSON_HERE: put your own kaggle.json in this folder
- Dockerfile: For containerization
- README.md: project description

#### Related works

- https://www.kaggle.com/datasets/payne18/road-detection-dataset-with-masks
- https://github.com/BME-SmartLab-VITMMA19/vision-assignment/blob/main/DL_Practice_4_Vision.ipynb

#### Run instructions

Prerequisites of running:

- docker
- kaggle registration

Before running the docker image, place your kaggle.json file in the data folder.

To build the docker image run the following command:

```
docker build -t vision-transformers .
```

To run the docker image run the following command:

```
docker run -p 8888:8888 -it vision-transformers
```

After running the docker image, you can open the jupiter notebook on http://127.0.0.1:8888/notebooks/main.ipynb and run the contents.

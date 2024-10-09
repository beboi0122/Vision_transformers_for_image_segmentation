kaggle datasets download -d payne18/road-detection-dataset-with-masks
mkdir ./images
unzip road-detection-dataset-with-masks -d ./images
python data_load.py
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
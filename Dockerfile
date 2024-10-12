# Use the official Jupyter base image with Python
FROM jupyter/minimal-notebook:python-3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1

COPY ./project .

RUN pip install -r requirements.txt

# Expose the default Jupyter port
EXPOSE 8888


CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
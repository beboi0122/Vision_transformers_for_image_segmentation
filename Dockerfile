# Use the official Jupyter base image with Python
FROM jupyter/minimal-notebook:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install any additional Python packages required (e.g., pandas, requests)
RUN pip install pandas requests kaggle

# Copy any notebooks into the container (optional)
COPY ./files .

#RUN chmod +x run.sh

# Expose the default Jupyter port
EXPOSE 8888

# Set the default command to start the Jupyter Notebook
CMD ["./run.sh"]
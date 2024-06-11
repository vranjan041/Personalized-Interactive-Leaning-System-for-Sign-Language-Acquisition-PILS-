# FROM python:3-alpine3.10
# WORKDIR /firstpart
# COPY . /firstpart
# RUN python -m pip install --upgrade pip
# RUN pip install -r requirements.txt
# EXPOSE 3000
# CMD python ./app.py

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     python-opencv

# Upgrade pip

FROM python:3.8

RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /firstpart

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6

# Copy application files
COPY . /firstpart

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 3000

# Command to run the application
CMD ["python", "./app.py"]

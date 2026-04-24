# Use a lightweight Python 3 base image with Debian Bullseye
FROM python:3.10-slim-bullseye

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y curl procps openjdk-11-jdk-headless && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

RUN pip install pandas \
    numpy \
    pyarrow \
    scikit-learn \
    scipy \
    matplotlib \
    seaborn \
    minio \
    hdfs \
    pyspark==3.5.1

# Set working directory
WORKDIR /pipeline
COPY . /pipeline

# Set default command
CMD ["python", "run_pipeline.py"]
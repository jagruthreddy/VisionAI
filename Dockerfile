# Use the official Python image from Docker Hub as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Update the package list and install necessary libraries for PyQt5 and OpenGL
RUN apt-get update
RUN apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-util1 \
    libxrender1 \
    libxi6 \
    libsm6 \
    libfontconfig1 \
    libfreetype6 \
    libxext6 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libxinerama1 \
    libxfixes3 \
    libxtst6 \
    libxss1 \
    libxxf86vm1

# Remove package lists to save space
RUN rm -rf /var/lib/apt/lists/*

# Set environment variables for Qt to use offscreen platform
ENV QT_QPA_PLATFORM=offscreen
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# Copy the requirements file into the container
COPY requirements.txt /usr/src/app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory contents into the container
COPY . /usr/src/app/

# Specify the command to run the application
CMD ["python", "./app.py"]
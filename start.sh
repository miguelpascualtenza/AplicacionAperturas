#!/bin/bash
set -e

# Instalar dependencias de compilación
apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
pip install --upgrade pip
pip install -r requirements.txt

# Ejecutar la aplicación
python app.py
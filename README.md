# vitCausalSeries
This is a vit (Vsion Transformer) project for treating time series data for image reconstruction and classification.
Use this for gdal installation.


https://stackoverflow.com/questions/44005694/no-module-named-gdal

sudo apt-get update && sudo apt upgrade -y && sudo apt autoremove 
sudo apt-get install -y cdo nco gdal-bin libgdal-dev


python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade gdal


conda install -c conda-forge libgdal
conda install -c conda-forge gdal
conda install tiledb=2.2
conda install poppler
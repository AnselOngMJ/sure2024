# POV24: Are Clouds Fractal?

**Supervisor Team**: Dr Adam Povey, Dr Kamil Mroz

**Categories**: Data Analysis

**Location**: Space Park Leicester

Clouds are ubiquitous on the Earth, covering about two-thirds of the planet at any given moment. They have numerous and conflicting impacts on climate by reflecting or absorbing solar and thermal radiation. Changes in the behaviour of clouds due to human activity is a major source of uncertainty in predictions of future climate. A cloud is smaller than the typical length scale of a pixel in a climate model, such that they must be represented by approximations rather than explicit physics. Climate models, therefore, make various assumptions about the shape, size, and evolution of clouds.

This student will use laser and radio ranging data from satellites and ground-sites to identify the vertical extent of clouds in a variety of environments. From that mask, the fractal dimension clouds will be estimated and compared to previous results. This may be used to assess the accuracy of recent climate modelling. Data analysis will be performed in Python and, though no prior coding experience is necessary, it would be beneficial.

## Installation

Create a conda virtual environment and install necessary packages and libraries:

```sh
conda create -y --name fractal-clouds-env
conda activate fractal-clouds-env
conda install -y -c conda-forge -c pytorch numpy pandas matplotlib basemap cartopy scipy scikit-image pytorch torchvision torchaudio cpuonly beautifulsoup4 netCDF4 pyhdf jupyter nb_conda_kernels ipywidgets ipykernel
pip install cloudnetpy opencv-python imutils
```

Run in JASMIN Notebooks shell:

```sh
conda run --name fractal-clouds-env python -m ipykernel install --user --name fractal-clouds-env
```

## Purpose of each file

`get_cloud_data.py` and `get_cloud_data.sh` are the main files for extracting clouds from the visualisations then storing the area, perimeter, and wind speed.

`fractal_plots.ipynb` is the main file for plotting the 2D histograms of fractal dimension against area/perimeter and also showing the types of clouds in different sections of the 2D histogram.

`convert.py` and `convert.sh` are files to convert `.nc` files downloaded from the ACTRIS Cloudnet data portal into `.png` visualisations.

`job_success_checker.py` is a helper script I wrote to check which jobs in the array failed and return the ranges that need to be retried.

`move.sh` is a helper script I wrote to move data from successful reruns into the previous job folder.

`cloud_fraction.ipynb`, `cloud_mask.ipynb`, and `radar_plots.ipynb` are files containing the code I wrote in the beginning weeks to understand the format of NetCDF and HDF files.

`watershed.ipynb` is my attempt at segmenting blobs of clouds.

`contour_visualiser.ipynb` has an interactive tool I wrote to visualise the contours found by my algorithm with a GUI to change the date and site.

## Scientific Functions

Some of the functions in `get_cloud_data.py` with no documentation can be explained with these equations from [IFS Documentation Part III - Chapter 2, Section 2.2.1](https://www.ecmwf.int/sites/default/files/elibrary/2023/81369-ifs-documentation-cy48r1-part-iii-dynamics-and-numerical-procedures.pdf) needed to calculate geopotential height on different levels.

```math
p_{k+1/2} = A_{k+1/2} + B_{k+1/2} p_{s}
```

```math
\Delta p_{k} = p_{k+1/2} - p_{k-1/2}
```

```math
T_{v} = T [1 + \{(R_{vap} / R_{dry}) - 1\} q - \sum_{k} q_{k}]
```

```math
\Phi_{k+1/2} = \Phi_{s} + \sum_{j=k+1}^{NLEV} R_{dry} (T_{v})_{j} ln\left(\frac{p_{j+1/2}}{p_{j-1/2}}\right)
```

```math
\alpha_{k} = 1 - \frac{p_{k-1/2}}{\Delta p_{k}} ln\left(\frac{p_{k+1/2}}{p_{k-1/2}}\right)
```

```math
\Phi_{k} = \Phi_{k+1/2} + \alpha_{k} R_{dry} (T_{v})_{k}
```

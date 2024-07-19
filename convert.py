import os

from cloudnetpy.plotting import generate_figure

num_files = 2000 # How many files to convert in one job
count = 0
png_folder = './cloudnet-collection'
nc_folder = './cloudnet-collection-nc'
png_files = set(os.listdir(png_folder))
nc_files = set(os.listdir(nc_folder))
for file in nc_files:
    if file.endswith('.nc') and file.removesuffix('.nc') + '.png' not in png_files:
        generate_figure(f'./{nc_folder}/{file}', ['target_classification'], output_filename=f'{png_folder}/' + file.removesuffix('.nc'), show=False)
        count += 1
        if count >= num_files:
            break

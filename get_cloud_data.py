import argparse
import math
import os
import dateutil
from datetime import timedelta

import cv2
import torch
import torchvision.ops.boxes as bops
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from netCDF4 import Dataset

class Era5:
    # Scientific constants
    R_VAP = 461.5
    R_DRY = 287.052874
    G = 9.80665
    RE = 6378.137

    # Image size constants
    HEIGHT, WIDTH = 371, 1245
    HEIGHT_PIXEL_RATIO = 12 / HEIGHT
    TIME_PIXEL_RATIO = 24 / WIDTH

    def __init__(self):
        self.ds_lnsp, self.ds_q, self.ds_t, self.ds_u, self.ds_v, self.ds_z = None, None, None, None, None, None
        self.clear()
        # Find values of constants a and b needed to calculate pressure on model levels
        # from the website table and put them in a dictionary
        self.ml_dict = {}
        with open('ml_defs.html') as fp:
            soup = BeautifulSoup(fp, 'html.parser')
        for tr in soup.tbody.find_all('tr'):
            td = tr.find_all('td')
            self.ml_dict[int(td[0].string)] = float(td[1].string), float(td[2].string)

    def open_files(self):
        self.ds_lnsp = get_dataset(self.path, 'lnsp', self.hour)
        self.ds_q = get_dataset(self.path, 'q', self.hour)
        self.ds_t = get_dataset(self.path, 't', self.hour)
        self.ds_u = get_dataset(self.path, 'u', self.hour)
        self.ds_v = get_dataset(self.path, 'v', self.hour)
        self.ds_z = get_dataset(self.path, 'z', self.hour)

    def close_files(self):
        """
        Close files to prevent too many files open error.
        """
        if self.ds_lnsp is not None:
            self.ds_lnsp.close()
            self.ds_q.close()
            self.ds_t.close()
            self.ds_u.close()
            self.ds_v.close()
            self.ds_z.close()
    
    def clear(self):
        """
        Resets dictionaries containing ERA5 variables that are used to cache data to reduce time
        opening files.
        """
        self.p_dict = {}
        self.q_dict = {}
        self.t_dict = {}
        self.u_dict = {}
        self.v_dict = {}
        self.geop_h_dict ={}

    def get_time_height_range(self, box):
        """
        Get the earliest/latest time and lowest/largest altitude that the cloud covers.

        Args:
            box: bounding box of the cloud

        Returns:
            time_range: earliest and latest time in hours
            height_range: lowest and largest altitude in kilometres
        """
        x, y, w, h = box
        time_range = round(Era5.TIME_PIXEL_RATIO * x), round(Era5.TIME_PIXEL_RATIO * (x + w))
        height_range = 12 - Era5.HEIGHT_PIXEL_RATIO * (y + h), 12 - Era5.HEIGHT_PIXEL_RATIO * y
        return time_range, height_range

    def get_pressure(self, ml, hour):
        if f'{hour}{ml}' not in self.p_dict:
            sp = np.exp(self.ds_lnsp['lnsp'][0, self.lat, self.lon]) / 100
            a_plus, b_plus = self.ml_dict[ml]
            p_plus = a_plus / 100 + b_plus * sp
            a_minus, b_minus = self.ml_dict[ml - 1]
            p_minus = a_minus / 100 + b_minus * sp
            self.p_dict[f'{hour}{ml}'] = p_plus, p_minus, p_plus - p_minus, p_plus / p_minus
        return self.p_dict[f'{hour}{ml}']
    
    def get_humidity(self, ml, hour):
        if f'{hour}{ml}' not in self.q_dict:
            self.q_dict[f'{hour}{ml}'] = self.ds_q['q'][0, ml - 1, self.lat, self.lon]
        return self.q_dict[f'{hour}{ml}']

    def get_temperature(self, ml, hour):
        if f'{hour}{ml}' not in self.t_dict:
            t = self.ds_t['t'][0, ml - 1, self.lat, self.lon]
            self.t_dict[f'{hour}{ml}'] = t * (1 + (Era5.R_VAP / Era5.R_DRY - 1) * self.get_humidity(ml, hour))
        return self.t_dict[f'{hour}{ml}']

    def get_alpha(self, ml, hour):
        if ml == 1:
            return math.log(2)
        elif ml > 1:
            _, p_minus, delta_p, p_divided = self.get_pressure(ml, hour)
            return 1 - p_minus / delta_p * math.log(p_divided)
    
    def get_geopotential(self, ml, hour):
        z_surface = self.ds_z['z'][0, self.lat, self.lon]
        z_half = z_surface + np.array([Era5.R_DRY * self.get_temperature(j, hour) * math.log(self.get_pressure(j, hour)[3]) for j in range(ml + 1, 138)]).sum()
        z = z_half + self.get_alpha(ml, hour) * Era5.R_DRY * self.get_temperature(ml, hour)
        return z
    
    def get_geopotential_height(self, ml, hour):
        return self.get_geopotential(ml, hour) / Era5.G / 1000
    
    def get_geometric_height(self, ml, hour):
        geop_height = self.get_geopotential_height(ml, hour)
        return Era5.RE * geop_height / (Era5.RE - geop_height)
    
    def get_ml_range(self, height_range, hour):
        min_h, max_h = height_range
        geop_h = self.geop_h_dict[hour]
        return abs(geop_h - max_h).argmin() + 1, abs(geop_h - min_h).argmin() + 1
    
    def get_wind_speed(self, contour):
        """
        Calculate average wind speed of a cloud across the altitudes and hours that it covers.

        Args:
            contour: contour of cloud

        Returns:
            wind_speed: average wind speed
        """
        time_range, height_range = self.get_time_height_range(cv2.boundingRect(contour))
        min_t, max_t = time_range
        min_h, max_h = height_range
        min_ml_dict = {}
        max_ml_dict = {}
        path_copy = self.path
        for i in range(min_t, max_t + 1):
            # Open a new file for the next day to get wind speeds for clouds the reach the 24 hour mark
            if i == 24:
                if 'era51' in self.path:
                    self.path = self.path.replace('era51', 'era5')
                new_date = (dateutil.parser.parse(self.path[66:78].replace('HH', '23')) +
                            timedelta(hours=1)).isoformat().replace('-', '').replace(':', '').replace('T', '')[:-6]
                # CEDA archive does not have complete ERA5 data from 2000-2006 so ERA5.1 is used
                if 2000 <= int(new_date[:4]) <= 2006:
                    if 'era51' in self.path:
                        self.path = self.path.replace('era51', 'era5')
                    self.path = self.path[:33] + new_date[:4] + self.path[37] + new_date[4:6] + \
                        self.path[40] + new_date[6:8] + self.path[43:66] + new_date + self.path[74:]
                    self.path = self.path.replace('era5', 'era51')
                else:
                    if 'era51' in self.path:
                        self.path = self.path.replace('era51', 'era5')
                    self.path = self.path[:33] + new_date[:4] + self.path[37] + new_date[4:6] + \
                        self.path[40] + new_date[6:8] + self.path[43:66] + new_date + self.path[74:]
            if i not in self.u_dict:
                geop_h = np.array([])
                self.hour = '0' + str(i) if i < 10 else str(i)
                hour_copy = self.hour
                if i == 24:
                    self.hour = '00'
                self.close_files()
                self.open_files()
                # Iterate over all model levels to get each geopotential height
                for j in range(1, 138):
                    geop_h = np.append(geop_h, self.get_geopotential_height(j, hour_copy))
                self.geop_h_dict[i] = geop_h
                self.u_dict[i] = self.ds_u['u'][0, :, self.lat, self.lon]
                self.v_dict[i] = self.ds_v['v'][0, :, self.lat, self.lon]
            min_ml, max_ml = self.get_ml_range([min_h, max_h], i)
            min_ml_dict[i] = min_ml
            max_ml_dict[i] = max_ml
        u = np.array([])
        v = np.array([])
        winds = []
        self.path = path_copy
        for i in range(min_t, max_t + 1):
            u = np.concatenate([u, self.u_dict[i][min_ml_dict[i] - 1:max_ml_dict[i]]])
            v = np.concatenate([v, self.v_dict[i][min_ml_dict[i] - 1:max_ml_dict[i]]])
            # Storing all wind speeds, time, and model levels as a string with separators
            winds.append('$'.join(['#'.join(self.u_dict[i][min_ml_dict[i] - 1:max_ml_dict[i]].astype(str)),
                                   '#'.join(self.v_dict[i][min_ml_dict[i] - 1:max_ml_dict[i]].astype(str)),
                                   str(i), str(min_ml_dict[i]), str(max_ml_dict[i])]))
        wind_speed = ((u ** 2 + v ** 2) ** 0.5).mean()
        return wind_speed, '@'.join(winds)

def get_scaled_perimeter(contour, x_ratio, y_ratio):
    """
    Gets an OpenCV contour and returns the perimeter of the contour based on how much distance each
    pixel represents in the x and y axes.
    Used to convert a contour perimeter from pixels to kilometres.

    Args:
        contour: an OpenCV contour of a cloud
        x_ratio: the width of one pixel in kilometres
        y_ratio: the height of one pixel in kilometres

    Returns:
        scaled_perimeter: the perimeter in kilometres
    """
    c = contour.astype(float)
    c[:, :, 0] *= x_ratio
    c[:, :, 1] *= y_ratio
    scaled_perimeter = 0
    for i in range(len(c) - 1):
        scaled_perimeter += np.linalg.norm(c[i] - c[i + 1])
    scaled_perimeter += np.linalg.norm(c[-1] - c[0])
    return scaled_perimeter

def intersect(box1, box2):
    """
    Determines whether an OpenCV bounding box intersects another bounding box.
    Used for determining when an ice cloud region is in contact with a rainy region
    and thus precipitating.

    Args:
        box1: first bounding box
        box2: second bounding box

    Returns:
        intersects: boolean on whether the boxes intersect
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    torch_box1 = torch.tensor(np.array([[x1, y1, x1 + w1, y1 + h1]]), dtype=torch.float)
    torch_box2 = torch.tensor(np.array([[x2, y2, x2 + w2, y2 + h2]]), dtype=torch.float)
    iou = bops.box_iou(torch_box1, torch_box2)
    return not not iou.any()

def longest_edge(contour):
    """
    Calculates the longest straight edge of a contour.
    Value can be used to determine whether a significant section of the cloud has been cut off
    by attenuation or border of the plot which affects how crinkly the perimeter is.

    Args:
        contour: contour around the cloud
    
    Returns:
        longest_edge: longest straight edge of the cloud
    """
    prev_diff = None
    length = 0
    max_length = 0
    for i in range(len(contour) - 1):
        diff = contour[i] - contour[i + 1]
        if (diff == prev_diff).all():
            length += 1
        else:
            length = 1
        prev_diff = diff
        max_length = max(length, max_length)
    return max_length

def get_dataset(path, var, hour):
    """
    Helper function to load a dataset from ERA5.

    Args:
        path: the path to the file to open
        var: which variable from ERA5 to get
        hour: the hour of the file from ERA5

    Returns:
        dataset: the dataset containing the variables needed in a specified hour
    """
    return Dataset(path.replace('HH', hour).replace('$', var), 'r')

def get_contours(img_hsv, img_gray, hsv_values):
    """
    Find contours by thresholding images with HSV values.

    Args:
        img_hsv: image in HSV format
        img_gray: grayscale image
        hsv_values: values to determine what colour to find contours around

    Returns:
        contours: contours found
    """
    mask = cv2.inRange(img_hsv, hsv_values, hsv_values)
    img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    img[img != 0] = 255
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return contours

def find_clouds(file_name):
    """
    Finds clouds that fulfill certain criteria which will be used to calculate fractal dimension.

    Args:
        file_name: name of image file to find clouds in

    Returns:
        contours_ice: contours of all valid ice clouds
    """
    img = cv2.imread(file_name)[39:410,72:1317,:]
    
    # Dark blue pixels that represent the classification of Ice/Droplets are reclassified as just Ice
    # instead to prevent random splitting of contours around ice clouds
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_drop = cv2.inRange(img_hsv, (119, 159, 185), (119, 159, 185))
    img[mask_drop> 0] = (187, 176, 160)
    
    # Image is converted to HSV as using hue to threshold images based on colour is easier
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    contours_ice = get_contours(img_hsv, img_gray, (102, 37, 187)) # Ice clouds
    contours_melt = get_contours(img_hsv, img_gray, (19, 255, 255)) # Melting ice
    contours_drizzle = get_contours(img_hsv, img_gray, (102, 221, 243)) # Rain

    contours_ice = [contour for contour in contours_ice if cv2.contourArea(contour) > 0]
    contours_melt = [contour for contour in contours_melt if cv2.contourArea(contour) > 0]
    contours_drizzle = [contour for contour in contours_drizzle if cv2.contourArea(contour) > 0]
    
    # Find bounding boxes for each contour to determine whether ice clouds are intersecting
    # with precipitating regions
    box_melt = []
    box_ice = []
    box_drizzle = []
    box_corrupt = []
    to_remove = []
    for contour in contours_ice:
        x, y, w, h = cv2.boundingRect(contour)
        if y < 3:
            box_corrupt.append([x, y, w, h])
        box_ice.append([x, y, w, h])
    
    for contour in contours_melt:
        x, y, w, h = cv2.boundingRect(contour)
        # Intersection method may not detect that ice clouds are precipitating
        # unless bounding box for precipitating regions are moved slightly up
        y -= 2
        box_melt.append([x, y, w, h])

    for contour in contours_drizzle:
        x, y, w, h = cv2.boundingRect(contour)
        y -= 2
        box_drizzle.append([x, y, w, h])
    
    for i, v in enumerate(box_ice):
        # Remove clouds that have a significantly long straight edge that may be caused by
        # attenuation or being outside the plot
        # Threshold of 0.3 for fraction of longest edge over perimeter can be fiddled with,
        # may not be optimal as I chose it after some quick visual inspections
        if longest_edge(contours_ice[i]) / cv2.arcLength(contours_ice[i], True) > 0.3:
            to_remove.append(i)
        else:
            # Remove clouds that are precipitating and 'clouds' in areas of the image where
            # data is unnatural
            for j in box_melt + box_drizzle + box_corrupt:
                if intersect(v, j):
                    to_remove.append(i)
                    break
    
    for i in sorted(to_remove, reverse=True):
        contours_ice.pop(i)
    
    return contours_ice

def main():
    """
    Iterates through all the files, finds contours and wind speeds for each cloud,
    saves them in a CSV file.
    """
    PATH_PREFIX = '/badc/ecmwf-era5/data/oper/an_ml/YYYY/MM/DD/ecmwf-era5_oper_an_ml_YYYYMMDDHH00.$.nc'
    # TODO: Add more sites if needed, latitude (N) and longitude (E)
    SITES_LAT_LON = {
        'chilbolton': (51.144, 358.561),
        'galati': (45.435, 28.037),
        'lindenberg': (52.208, 14.118),
        'munich': (48.148, 11.573),
    }

    era5 = Era5()
    clouds = []
    count = 0

    # TODO: You can change how many files each array job processes
    NUM_FILES = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int)
    parser.add_argument('-d', '--directory')
    args = parser.parse_args()
    skip = 0 # Used to assist array jobs in selecting different files to process

    for file in os.listdir('./cloudnet-collection'):
        era5.clear()
        year, month, day = file[:4], file[4:6], file[6:8]

        # TODO: More recent files can be used once ERA5 dataset catches up
        if int(year) >= 2024 and int(month) >= 4:
            continue

        if skip < args.count * NUM_FILES:
            skip += 1
            continue
        
        path = PATH_PREFIX.replace('YYYY', year).replace('MM', month).replace('DD', day)
        if 2000 <= int(year) <= 2006:
            if 'era51' not in path:
                path = path.replace('era5', 'era51')
        era5.path = path

        site = file.split('_')[1]
        era5.lat = 360 - round(SITES_LAT_LON[site][0] * 4)
        era5.lon = round(SITES_LAT_LON[site][1] * 4)
        
        contours = find_clouds('./cloudnet-collection/' + file)

        for contour in contours:
            wind_speed, winds = era5.get_wind_speed(contour)
            length_pixel_ratio = wind_speed * Era5.TIME_PIXEL_RATIO * 3600 / 1000
            cloud = [
                cv2.contourArea(contour), cv2.arcLength(contour, True),
                cv2.contourArea(contour) * length_pixel_ratio * Era5.HEIGHT_PIXEL_RATIO,
                get_scaled_perimeter(contour, length_pixel_ratio, Era5.HEIGHT_PIXEL_RATIO),
                cv2.boundingRect(contour)[2], cv2.boundingRect(contour)[2] * length_pixel_ratio,
                wind_speed, winds, file
            ]
            clouds += [cloud]
        count += 1

        if count >= NUM_FILES:
            break

    era5.close_files()
    clouds = np.array(clouds)
    df = pd.DataFrame(clouds, columns=['area_px', 'perimeter_px', 'area_km', 'perimeter_km',
                                        'duration_px', 'length_km', 'wind_speed', 'winds', 'file_name'])
    df.to_csv(args.directory + '/' + str(args.count))

if __name__ == '__main__':
    main()

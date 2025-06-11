import os
import sys
import pylab
import numpy as np
from astropy.io import fits
from lmfit import minimize, Parameters, Parameter
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import math
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.optimize import curve_fit


## load the image and will plot the image
hdulist=fits.open('C:/Users/Rahi/Desktop/Galaxy Data/DDO 168/frame-u-003699-2-0084.fits.bz2')  ##repalce with your image
ra_image = 198.6133333
dec_image = 45.9294444
psf_image = 2.08
cdelt_image = 0.00010997562836385767
image_data=hdulist[0].data
hdu = hdulist[0]
data = hdulist[0].data
plt.figure()
plt.imshow(image_data, vmin=0, vmax=0.25)
plt.colorbar()


## calculate the SD and noise
dyn_range=(np.max(image_data)-np.min(image_data))

median = np.median(image_data)
new_image_data = image_data - median.flatten()
below_median = new_image_data < median
noise = new_image_data[below_median]

y, x = np.histogram(noise, bins=10, density=True)

dx = x[1] - x[0]
x = x[:-1] + dx/2

def gauss1(dparams,x,data): # Single component
    a1 = dparams['amp'].value
    b1 = dparams['loc'].value
    c1 = dparams['sig'].value
    model = a1*np.exp(-((x-b1)**2.0)/(2.0*c1**2))
    return data - model

zmax = np.max(y)
below_median = np.where(y < zmax*0.9)
tx = x[below_median][-1]
below_median = np.where(y == zmax)
ttx = x[below_median][0]
zloc = ttx*1
zsig = (ttx - tx)

params = Parameters()
params.add('amp',value=zmax)
params.add('loc', value= zloc)
params.add('sig', value= zsig)

mi = minimize(gauss1, params, args=(x,y), method='least_square')
res = mi.residual
convgd = int(mi.errorbars)
v=[mi.params['amp'].value,mi.params['loc'].value,mi.params['sig'].value]
verr=[mi.params['amp'].stderr,mi.params['loc'].stderr,mi.params['sig'].stderr]
rv = v[0]*np.exp(-((x-v[1])**2.0)/(2.0*v[2]**2))


## will find the stars using gaia and plot them
image_shape = hdu.data.shape
wcs = WCS(hdu.header)
coord = wcs.pixel_to_world(0,2*wcs.wcs.crpix[1])
c1 = SkyCoord(coord.ra,coord.dec, frame='icrs',unit='deg')
c2 = SkyCoord(ra_image,dec_image, frame='icrs',unit='deg')   
sep = c1.separation(c2)
coord = SkyCoord(ra=ra_image, dec=dec_image, unit=(u.degree, u.degree), frame='icrs')   
job = Gaia.cone_search_async(coord, radius=u.Quantity(sep.deg, u.deg))
result = job.get_results()
table = Table()
table['ra'] = result['ra']
table['dec'] = result['dec']
table['magnitude'] = result['phot_bp_mean_mag']
sorted_table = table[np.argsort(table['magnitude'])]
sorted_ra = sorted_table['ra']
sorted_dec = sorted_table['dec']
#sorted_magnitude = sorted_table['phot_bp_mean_mag']
ra = result['ra']
dec = result['dec']
c = SkyCoord(ra,dec, frame='icrs',unit='deg')
xpixels=[]
ypixels=[]
x_pixel, y_pixel = wcs.world_to_pixel(c) 
for i in range (len(x_pixel)):
    if (x_pixel[i] < image_shape[1]):
        if (y_pixel[i] < image_shape[0]):
            xpixels.append(x_pixel[i])
            ypixels.append(y_pixel[i])

xpixel = np.array(xpixels)
ypixel = np.array(ypixels)
# plt.figure()
# plt.imshow(image_data, vmin=0, vmax=0.25)

x_positions = (xpixel-1) 
y_positions = (ypixel-1)
# plt.plot(xpixel, ypixel, marker='o', color='None', mec='r', mew=2, ms=12)
# plt.show()

## will plot the stars with a provided SNR
new_x =[]
new_y=[]
def max_pixel_value(a,b):
    radius = 3
    pixel_positions = []
    for y in range(int(b - radius), int(b + radius+1)):
        for x in range(int(a - radius), int(a + radius+1)):
            if (x >= 0 and x < image_shape[1]) and (y >= 0 and y < image_shape[0]) and ((x - a) ** 2 + (y - b) ** 2) <= radius ** 2:

                pixel_value = image_data[y, x]
                pixel_positions.append(pixel_value)
    star_max = np.max(pixel_positions)
    return star_max
    
for i in range(len(xpixel)):
    output = max_pixel_value(xpixel[i], ypixel[i])
    SNR = output / v[2] 
    if SNR >5:
        new_x.append(xpixel[i])
        new_y.append(ypixel[i])
new_x_array = np.array (new_x)
new_y_array = np.array (new_y)

plt.figure()
plt.imshow(image_data, vmin=0, vmax=0.25)
new_x_pos=[]
new_y_pos=[]
for nx in new_x:
        new_x_pos.append(nx-1)
for ny in new_y:
        new_y_pos.append(ny-1)
plt.plot(new_x, new_y, marker='o', color='None', mec='r', mew=2, ms=12)    
plt.colorbar()
plt.show()


## gaussian plot of each star
def gaussian_check(a,b):
    box_size = 15
    box_x = np.arange(int(a) - box_size // 2, int(a) + box_size // 2)
    box_y = np.arange(int(b) - box_size // 2, int(b) + box_size // 2)
    box_data = data[box_y[:, np.newaxis], box_x]
    #print (box_data)

    # Define the 2D Gaussian function to fit
    def gaussian_2d(xy_mesh, amplitude, x_mean, y_mean, x_stddev, y_stddev):
        x, y = xy_mesh
        return amplitude * np.exp(-((x - x_mean) ** 2 / (2 * x_stddev ** 2) + (y - y_mean) ** 2 / (2 * y_stddev ** 2)))

    # Fit the 2D Gaussian curve to the pixel values
    x_mesh, y_mesh = np.meshgrid(box_x, box_y)
    x_mesh_flat = x_mesh.flatten()
    y_mesh_flat = y_mesh.flatten()
    box_data_flat = box_data.flatten()
    initial_guess = [np.max(box_data), box_x.mean(), box_y.mean(), box_x.std(), box_y.std()]
    fit_params, _ = curve_fit(gaussian_2d, (x_mesh_flat, y_mesh_flat), box_data_flat, p0=initial_guess,maxfev=5000000)

    # Generate the curve using the fitted parameters
    fit_data = gaussian_2d((x_mesh, y_mesh), *fit_params)
    gaussian_center_x = fit_params[1]
    gaussian_center_y = fit_params[2]

    dist = np.sqrt((gaussian_center_x - a)**2 + (gaussian_center_y - b)**2)
    return dist
    
for i in range(len(new_x_array)):
    output = gaussian_check(new_x_array[i], new_y_array[i])
    
    
### code for masking
mask_x =[]
mask_y=[]
sdss_image_file = 'C:/Users/Rahi/Desktop/Galaxy Data/DDO 168/frame-u-003699-2-0084.fits.bz2'  # Provide the path to your SDSS image FITS file
#print ("start")
def psf_check(a,b,image_file):
    def calculate_psf(sdss_image_file, psf_width, star_coordinates):
        hdulist = fits.open(sdss_image_file)
        image_data = hdulist[0].data
        def gaussian_check(a,b):
            box_size = 15
            box_x = np.arange(int(a) - box_size // 2, int(a) + box_size // 2)
            box_y = np.arange(int(b) - box_size // 2, int(b) + box_size // 2)
            box_data = data[box_y[:, np.newaxis], box_x]
            #print (box_data)

            # Define the 2D Gaussian function to fit
            def gaussian_2d(xy_mesh, amplitude, x_mean, y_mean, x_stddev, y_stddev):
                x, y = xy_mesh
                return amplitude * np.exp(-((x - x_mean) ** 2 / (2 * x_stddev ** 2) + (y - y_mean) ** 2 / (2 * y_stddev ** 2)))

            # Fit the 2D Gaussian curve to the pixel values
            x_mesh, y_mesh = np.meshgrid(box_x, box_y)
            x_mesh_flat = x_mesh.flatten()
            y_mesh_flat = y_mesh.flatten()
            box_data_flat = box_data.flatten()
            initial_guess = [np.max(box_data), box_x.mean(), box_y.mean(), box_x.std(), box_y.std()]
            fit_params, _ = curve_fit(gaussian_2d, (x_mesh_flat, y_mesh_flat), box_data_flat, p0=initial_guess,maxfev=500000)

            # Generate the curve using the fitted parameters
            fit_data = gaussian_2d((x_mesh, y_mesh), *fit_params)
            gaussian_center_x = fit_params[1]
            gaussian_center_y = fit_params[2]

            dist = np.sqrt((gaussian_center_x - a)**2 + (gaussian_center_y - b)**2)
            return dist

        # Get the coordinates of the star
        star_x, star_y = star_coordinates

        # Convert star coordinates to integers
        star_x = int(star_x)
        star_y = int(star_y)

        # Extract the region around the star
        star_size = 15  # Adjust the size as needed
        star_image = image_data[star_y - star_size:star_y + star_size, star_x - star_size:star_x + star_size]

        output1 = gaussian_check(new_x_array[i], new_y_array[i])
        psf = psf_width
        if (output1 < (psf/(3600*cdelt_image) )):  

            star_x = a
            star_y = b

            radius = 8
            pixel_values1 = []
            pixel_x=[]
            pixel_y=[]

            for y in range(int(star_y - radius), int(star_y + radius + 1)):
                for x in range(int(star_x - radius), int(star_x + radius + 1)):
                # Check if the pixel is within the circular region
                    if ((x - star_x) ** 2 + (y - star_y) ** 2) <= radius ** 2:
                        pixel_value1 = image_data[y, x]
                        pixel_values1.append(pixel_value1)
                        pixel_x.append(x)
                        pixel_y.append(y)
                    #print (x,y)

            x_pixels=np.array(pixel_x)
            y_pixels=np.array(pixel_y)
            hdulist = fits.open(sdss_image_file)
            image_data = hdulist[0].data

            # Calculate noise statistics from the image or a specific region of interest
            mean_noise = v[1]  # Replace with the calculated mean noise value
            std_noise = v[2]   # Replace with the calculated standard deviation of the noise

            # Generate random noise values based on the statistics
            noise = np.random.normal(loc=mean_noise, scale=std_noise, size=len(x_pixels))

            # Identify the pixels where you want to replace the values with noise
            noise_pixels = np.column_stack((x_pixels, y_pixels))

            # Replace the identified pixel values with the generated noise values
            for pixel_idx, pixel in enumerate(noise_pixels):
                x, y = pixel
                image_data[y, x] = noise[pixel_idx]
        return psf,image_data

    hdulist = fits.open(image_file)
    image_data = hdulist[0].data    
    psf_width_u = psf_image  # Provide the effective PSF width from 2-Gaussian fit in the u-band
    star_coordinates = (a, b)  
    
    out1,sdss_new= calculate_psf(image_file, psf_width_u, star_coordinates)
    return out1,sdss_new

for i in range(len(new_x_array)):
    if i == 0:
        final_out,sdss_image_new1 = psf_check(new_x_array[i],new_y_array[i],sdss_image_file)
        modified_image_data = sdss_image_new1 
        hdul1 = fits.PrimaryHDU(modified_image_data)
        output_filename = f'C:/Users/Rahi/Desktop/testing2/modified_image_data_{i}.fits'   ## replace with the path where you want to save the masked images
        hdul1 = fits.PrimaryHDU(modified_image_data)
        hdul1.writeto(output_filename, overwrite=True)
        
    if i > 0:
        input_filename = f'C:/Users/Rahi/Desktop/testing2/modified_image_data_{i-1}.fits'  ## replace with the path where you want to save the masked images
        sdss_image_new = input_filename  # Provide the path to your SDSS image FITS file
        final_out,sdss_image_new1 = psf_check(new_x_array[i],new_y_array[i],sdss_image_new)
        modified_image_data = sdss_image_new1 
        hdul1 = fits.PrimaryHDU(modified_image_data)
        output_filename = f'C:/Users/Rahi/Desktop/testing2/modified_image_data_{i}.fits'   ## replace with the path where you want to save the masked images
        hdul1 = fits.PrimaryHDU(modified_image_data)
        hdul1.writeto(output_filename, overwrite=True)
for k in range (len(new_x_array)):
    if k == (len(new_x_array)-1):
        output_filename = f'C:/Users/Rahi/Desktop/testing2/modified_image_data_{k}.fits'  ## replace with the path where you want to save the masked images
        hdulist=fits.open(output_filename)
        final_modified_data=hdulist[0].data
        plt.figure()
        plt.imshow(final_modified_data,vmin=0,vmax=0.25)
        plt.show()





#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
	float sum = 0;
    int img_size = im.w * im.h * im.c;
	
	for (int idx=0; idx < img_size; idx++){
		sum += im.data[idx];
	}
	
	for (int idx=0; idx < img_size; idx++){
		im.data[idx] /= sum;
	} 
}

image make_box_filter(int w)
{
    image filter = make_image(w,w,1);
	float v = (float)1 / (w * w);
	
	for (int idx=0; idx < w*w; idx++){
		filter.data[idx] = v;
	} 
    return filter;
}

//One function for each convolution type, for readability
// 1: im.c == filter.c, output image with c channels 
// 2: filter.c == 1, output image with c channels 
// 1: im.c == filter.c, merge channels 
// 2: filter.c == 1, merge channels

float convolve_type_1(image im, image filter, int x_start, int y_start, int c){
	float sum = 0;
	int col_offset = filter.w / 2 - x_start;
	int row_offset = filter.h / 2 - y_start;
	
	for (int col=0; col < filter.w; col++){
		for (int row=0; row < filter.h; row++){
			sum += get_pixel(im, col-col_offset, row-row_offset, c) * get_pixel(filter, col, row, c);
		}
	}	
	return sum;
}

float convolve_type_2(image im, image filter, int x_start, int y_start, int c){
	float sum = 0;
	int col_offset = filter.w / 2 - x_start;
	int row_offset = filter.h / 2 - y_start;
	
	for (int col=0; col < filter.w; col++){
		for (int row=0; row < filter.h; row++){
			sum += get_pixel(im, col-col_offset, row-row_offset, c) * get_pixel(filter, col, row, 0);
		}
	}	
	return sum;
}

float convolve_type_3(image im, image filter, int x_start, int y_start){
	float sum = 0;
	int col_offset = filter.w / 2 - x_start;
	int row_offset = filter.h / 2 - y_start;
	
	for (int col=0; col < filter.w; col++){
		for (int row=0; row < filter.h; row++){
			for (int c=0; c < im.c; c++){
				sum += get_pixel(im, col-col_offset, row-row_offset, c) * get_pixel(filter, col, row, c);
			}
		}
	}	
	return sum;
}

float convolve_type_4(image im, image filter, int x_start, int y_start){
	float sum = 0;
	int col_offset = filter.w / 2 - x_start;
	int row_offset = filter.h / 2 - y_start;
	
	for (int col=0; col < filter.w; col++){
		for (int row=0; row < filter.h; row++){
			for (int c=0; c < im.c; c++){
				sum += get_pixel(im, col-col_offset, row-row_offset, c) * get_pixel(filter, col, row, 0);
			}
		}
	}	
	return sum;
}

image convolve_image(image im, image filter, int preserve)
{
    assert(im.c == filter.c || filter.c == 1);
	image filtered;
	
	if (preserve == 1){
		filtered = make_image(im.w, im.h, im.c);
		if (im.c == filter.c){
			for (int col=0; col < im.w; col++){
				for (int row=0; row < im.h; row++){
					for (int c=0; c < im.c; c++){
						float v = convolve_type_1(im, filter, col, row, c);
						set_pixel(filtered, col, row, c, v);
					}
				}
			}
			
		} else if (filter.c == 1){
			for (int col=0; col < im.w; col++){
				for (int row=0; row < im.h; row++){
					for (int c=0; c < im.c; c++){
						float v = convolve_type_2(im, filter, col, row, c);
						set_pixel(filtered, col, row, c, v);
					}
				}
			}
		} 
	} else {
		filtered = make_image(im.w, im.h, 1);
		if (im.c == filter.c){
			for (int col=0; col < im.w; col++){
				for (int row=0; row < im.h; row++){
					float v = convolve_type_3(im, filter, col, row);
					set_pixel(filtered, col, row, 0, v);
				}
			}
		} else if (filter.c == 1){
			for (int col=0; col < im.w; col++){
				for (int row=0; row < im.h; row++){
					float v = convolve_type_4(im, filter, col, row);
					set_pixel(filtered, col, row, 0, v);
				}
			}
		} 
	}
    return filtered;
}

float get_gauss_val(float sigma, float x, float y){
	return exp(-1 * (pow(x,2) + pow(y,2))/(2*pow(sigma,2)))/(2 * M_PI * pow(sigma,2));
}

image make_gaussian_filter(float sigma)
{
    int size = sigma * 6 + 1;
	size = (size % 2 == 1) ? size : size+1;
	int radius = size / 2;
	image filter = make_image(size,size,1);
	
	for (int col=0; col < size; col++){
		for (int row=0; row < size; row++){
			set_pixel(filter, col, row, 0, get_gauss_val(sigma, col-radius, row-radius));
		}
	}
	l1_normalize(filter);
    return filter;
}


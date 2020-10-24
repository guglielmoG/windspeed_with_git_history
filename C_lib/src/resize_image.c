#include <math.h>
#include <stdio.h>
#include "image.h"

// int round(float x){
	// int truncated = (int)x;
	// return ((x - truncated) >= 0.5) ? truncated +1 : truncated;
// }

float nn_interpolate(image im, float x, float y, int c)
{
    return get_pixel(im, round(x), round(y), c);
}

//slow implementation, using functions previously defined
//slow because always computing memory index, could just traverse the memory directly but then need do custom code
image nn_resize(image im, int w, int h)
{
	//create new image, will contine resized final version
    image res_im = make_image(w,h,im.c);
	
	//compute transformation parameters Y = aX + b
	//X is a coordinate in new image, Y in old one. One set of params per dimension
	float a_w = (float)im.w / w;
	float b_w = 0.5 * (a_w - 1);
	
	float a_h = (float)im.h / h;
	float b_h = 0.5 * (a_h - 1);
	
    for (int row=0; row < h; row ++){
		for (int col=0; col < w; col ++){
			for (int c=0; c < im.c; c ++){
				float old_row = a_h * row + b_h;
				float old_col = a_w * col + b_w;
				float v = nn_interpolate(im, old_col, old_row, c);
				set_pixel(res_im, col, row, c, v);
			}
		}
	}
	
    return res_im;
}


float bilinear_interpolate(image im, float x, float y, int c)
{
	//identify coordinates of the box
	int x_0, x_1, y_0, y_1;
	if (x >= 0){
		x_0 = (int)x;
		x_1 = (int)x + 1;
	} else {
		x_0 = -1;
		x_1 = 0;
	}
	
	if (y >= 0){
		y_0 = (int)y;
		y_1 = (int)y + 1;
	} else {
		y_0 = -1;
		y_1 = 0;
	}
	
	//distances between middle point and corners (manhattan)
	float m = y - y_0;
	float l = x - x_0;
	
	//get values of corner pixels. bl = bottom left
	float bl = get_pixel(im, x_0, y_1, c);
	float br = get_pixel(im, x_1, y_1, c);
	float tl = get_pixel(im, x_0, y_0, c);
	float tr = get_pixel(im, x_1, y_0, c);
	
    //return m*(bl*(1-l) + br*l) + (1-m)*(tr*l + tl*(1-l));
	return (1-l)*(tl*(1-m)+bl*m) + l*(tr*(1-m)+br*m);
}

image bilinear_resize(image im, int w, int h)
{
    //create new image, will contine resized final version
    image res_im = make_image(w,h,im.c);
	
	//compute transformation parameters Y = aX + b
	//X is a coordinate in new image, Y in old one. One set of params per dimension
	float a_w = (float)im.w / w;
	float b_w = 0.5 * (a_w - 1);
	
	float a_h = (float)im.h / h;
	float b_h = 0.5 * (a_h - 1);
	
    for (int row=0; row < h; row ++){
		for (int col=0; col < w; col ++){
			for (int c=0; c < im.c; c ++){
				float old_row = a_h * row + b_h;
				float old_col = a_w * col + b_w;
				float v = bilinear_interpolate(im, old_col, old_row, c);
				set_pixel(res_im, col, row, c, v);
			}
		}
	}
	
    return res_im;
}


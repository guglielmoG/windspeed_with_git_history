#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

int get_position(image im, int x, int y, int c){
	return c * (im.w * im.h) + y * im.w + x;
}

float get_pixel(image im, int x, int y, int c)
{
    if (x < 0){ x = 0;}
	if (x >= im.w){ x = im.w-1;}
	if (y < 0){ y = 0;}
	if (y >= im.h){ y = im.h-1;}
	int idx = get_position(im, x, y, c);
    return im.data[idx];
}


//x = col, y = row, c = channel
void set_pixel(image im, int x, int y, int c, float v)
{
    if (!(x >= im.w || y >= im.h || c >= im.c || x < 0 || y < 0 || c < 0 )){
		int idx = get_position(im, x, y, c);
		im.data[idx] = v;
	}
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, im.w*im.h*im.c*sizeof(float));
    return copy;
}


void shift_image(image im, int c, float v)
{
    int img_size = im.w*im.h;
    for (int idx=img_size*c; idx < img_size*(c+1); idx ++){
		im.data[idx] += v;
	}
}

void clamp_image(image im)
{
    for (int idx = 0; idx < im.w*im.h*im.c; idx++){
		if (im.data[idx] > 1){im.data[idx]=1;}
		if (im.data[idx] < 0){im.data[idx]=0;}
	}
}

void scale_image(image im, int c, float v){
	int img_size = im.w*im.h;
    for (int idx=img_size*c; idx < img_size*(c+1); idx ++){
		im.data[idx] *= v;
	}
}

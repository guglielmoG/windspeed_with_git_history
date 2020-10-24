#ifndef IMAGE_H
#define IMAGE_H


typedef struct{
    int w,h,c;
    float *data;
} image;

// Basic operations
float get_pixel(image im, int x, int y, int c);
void set_pixel(image im, int x, int y, int c, float v);
image copy_image(image im);
void shift_image(image im, int c, float v);
void scale_image(image im, int c, float v);
void clamp_image(image im);


// Loading and saving
image make_image(int w, int h, int c);
image load_image(char *filename);
void save_image(image im, const char *name);
void save_png(image im, const char *name);
void free_image(image im);

// Resizing
float nn_interpolate(image im, float x, float y, int c);
image nn_resize(image im, int w, int h);
float bilinear_interpolate(image im, float x, float y, int c);
image bilinear_resize(image im, int w, int h);

// Filtering
image convolve_image(image im, image filter, int preserve);
image make_box_filter(int w);
image make_gaussian_filter(float sigma);

#endif


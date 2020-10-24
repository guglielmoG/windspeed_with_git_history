from my_lib_cv import *
import math 


#2
im = load_image("dog.jpg")
a = nn_resize(im, im.w//7, im.h//7)
save_image(a, "dog4x-bl")

im = load_image("dog.jpg")
a = bilinear_resize(im, im.w//7, im.h//7)
save_image(a, "dog4x-bl_0")

im = load_image("dog.jpg")
f = make_box_filter(7)
blur = convolve_image(im, f, 1)
a = nn_resize(blur, blur.w//7, blur.h//7)
save_image(a, "dog4x-bl_1")

im = load_image("dog.jpg")
f = make_box_filter(7)
blur = convolve_image(im, f, 1)
a = bilinear_resize(blur, blur.w//7, blur.h//7)
save_image(a, "dog4x-bl_2")

im = load_image("dog.jpg")
f = make_gaussian_filter(2)
blur = convolve_image(im, f, 1)
a = nn_resize(blur, blur.w//7, blur.h//7)
save_image(a, "dog4x-bl_3")

im = load_image("dog.jpg")
f = make_gaussian_filter(2)
blur = convolve_image(im, f, 1)
a = bilinear_resize(blur, blur.w//7, blur.h//7)
save_image(a, "dog4x-bl_4")


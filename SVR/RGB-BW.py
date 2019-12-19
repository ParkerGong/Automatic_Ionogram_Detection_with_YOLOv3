# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:58:39 2019

@author: 公园
"""

from PIL import Image
image_file = Image.open("20130408140700(cut).jpg") # open colour image
image_file = image_file.convert('1') # convert image to black and white
image_file.save('result.png')
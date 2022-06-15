import seam_carving
from os import startfile
from typing import OrderedDict
from numpy.lib.function_base import copy
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from PIL import ImageTk, Image
import tkinter as tk
SIZE_w = 1200
SIZE_h = 1080


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def importImg(label):
    global img_input
    global current
    current = None
    filetypes = (
        ('all files', '*.*'),
        ('png files', '*.png'),
        ('jpg files', '*.jpg')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='C:/Users/THAI NGUYEN/Desktop/BaoCao_CS406/img',
        filetypes=filetypes
    )

    if not filename:
        showinfo(
            title='Error!',
            message='Please import again'
        )
        return

    img_input = cv2.imread(filename)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input_display = img_input.copy()

    global h, w
    h, w = img_input_display.shape[:2]
    if h > SIZE_h:
        img_input_display = image_resize(img_input_display, height=SIZE_h)
    if w > SIZE_w:
        img_input_display = image_resize(img_input, width=SIZE_w)
    h, w = img_input_display.shape[:2]

    img_1 = ImageTk.PhotoImage(image=Image.fromarray(img_input_display))

    label.config(image=img_1)
    label.image = img_1
    label.place(x=1, y=1)


def exportImg(img):
    defaultextension = '.png'
    filename = fd.asksaveasfile(mode='w', defaultextension=defaultextension)
    if not filename:
        return
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename.name, img)


def getValue():
    msg = f'You entered  {height_input.get()} and  {width_input.get()} and {energy_input.get()} and {first_option.get()} and {isKeepMask.get()} '
    showinfo(
        title='Information',
        message=msg
    )


current = None
window = Tk()
window.geometry('1550x763')
window.state('zoomed')
window.resizable(0, 0)
window.title('Resize')

my_label = Button(window)

import_button = Button(window, text='Import image', bg='black', fg='white')
import_button.config(command=lambda: importImg(my_label))
import_button.config(height=2, width=15)
import_button.place(x=25, y=705)

height_input = ttk.Entry(window, width=15)
# height_input.insert(0, h)
width_input = ttk.Entry(window, width=15)
width_input.place(x=1380, y=105)
height_input.place(x=1380, y=75)


energy_input = ttk.Combobox(window, width=15)

# Adding combobox drop down list
energy_input['values'] = ('backward',
                          'forward')
energy_input.current()
energy_input.place(x=1380, y=135)


first_option = ttk.Combobox(window, width=15)
first_option['values'] = ('width-first',
                          'height-first',
                          'optimal')
first_option.current()
first_option.place(x=1380, y=165)

# keep mask


def showKeepMask():
    global keep_mask
    keep_mask = None


isKeepMask = BooleanVar()
is_keep_mask = Checkbutton(
    window, text='Chọn đối tượng giữ', variable=isKeepMask, command=showKeepMask)
is_keep_mask.place(x=1380, y=195)
# test

btn_getvalue = Button(window, text='get Value', bg='black', fg='white')
btn_getvalue.config(command=getValue)
btn_getvalue.config(height=2, width=15)
btn_getvalue.place(x=25, y=750)

# run


def seam_carving_resize():
    global height
    global width
    global energy
    global order
    height = int(height_input.get())
    width = int(width_input.get())
    energy = energy_input.get()
    order = first_option.get()

    img_output = seam_carving.resize(
        img_input, (width, height), energy, order, keep_mask=None)


global img_output
btn_getvalue = Button(window, text='run', bg='black',
                      fg='white')
btn_getvalue.config(command=seam_carving_resize)
btn_getvalue.config(height=2, width=15)
btn_getvalue.place(x=1380, y=300)
# export
btn_getvalue = Button(window, text='export', bg='black', fg='white')
btn_getvalue.config(command=lambda: exportImg(img_output))
btn_getvalue.config(height=2, width=15)
btn_getvalue.place(x=1380, y=800)


window.mainloop()

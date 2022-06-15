from os import startfile
from numpy.lib.function_base import copy
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from PIL import ImageTk, Image


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
        initialdir='./img',
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


current = None
window = Tk()
window.geometry('1550x763')
window.state('zoomed')
window.resizable(0, 0)
window.title('Some Photoshop filters')

my_label = Button(window)

import_button = Button(window, text='Import image', bg='black', fg='white')
import_button.config(command=lambda: importImg(my_label))
import_button.config(height=2, width=15)
import_button.place(x=25, y=705)
# dropdown options
OPTIONS = [
    "Resize",
    "Remove Object"
]


height_input = ttk.Entry(window, width=15)
width_input = ttk.Entry(window, width=15)
width_input.place(x=1420, y=105)
height_input.place(x=1420, y=75)
height_input.forget()
width_input.forget()


def callbackFunc(event):
    selected_value = event.widget.get()
    if selected_value == OPTIONS[0]:
        height_input.pack()
        width_input.pack()

    else:
        height_input.forget()
        width_input.forget()


option = ttk.Combobox(window, value=OPTIONS)
option.place(x=1420, y=45)
option.current(0)
option.config(height=50, width=15)
option.bind("<<ComboboxSelected>>", callbackFunc)


window.mainloop()

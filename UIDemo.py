from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
from PIL import ImageTk, Image
import seam_carving
import numpy as np

SIZE_w = 1400
SIZE_h = 720

pts = []
src_w = 0
src_h = 0
keep_mask = None
drop_mask = None
current = "remove"


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def importImg(label):
    global img_input
    filetypes = (
        ('all files', '*.*'),
        ('png files', '*.png'),
        ('jpg files', '*.jpg')
    )
    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='./',
        filetypes=filetypes
    )
    if not filename:
        showinfo(
            title='Error!',
            message='Please import again'
        )
        return
    img_input = cv2.imread(filename)
    img_input_display = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    h, w = img_input_display.shape[:2]
    if h > SIZE_h:
        img_input_display = image_resize(img_input_display, height=SIZE_h)
    h, w = img_input_display.shape[:2]
    if w > SIZE_w:
        img_input_display = image_resize(img_input_display, width=SIZE_w)
    h, w = img_input_display.shape[:2]
    img_1 = ImageTk.PhotoImage(image=Image.fromarray(img_input_display))
    label.config(image=img_1)
    label.image = img_1
    label.place(x=1, y=1)
    global src_w
    global src_h
    src_h, src_w = img_input.shape[:2]
    height_input.delete(0, END)
    width_input.delete(0, END)
    height_input.insert(0, src_h)
    width_input.insert(0, src_w)


def exportImg(img):
    defaultextension = '.png'
    filename = fd.asksaveasfile(mode='w', defaultextension=defaultextension)
    if not filename:
        return
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename.name, img)


def draw_roi(event, x, y, flags, param):
    img2 = img_input.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        pts.pop()
    if event == cv2.EVENT_MBUTTONDOWN:
        mask = np.zeros(img2.shape[:2], np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))
        show_image = cv2.addWeighted(
            src1=img2, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)
        ROI = cv2.bitwise_and(mask2, img2)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)
    if len(pts) > 0:
        cv2.circle(img2, pts[-1], 1, (0, 0, 255), -1)
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 1, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1],
                     color=(255, 0, 0), thickness=2)
    cv2.imshow('mask_', img2)


def drawMask(opt):
    img = img_input.copy()
    cv2.namedWindow('mask_')
    cv2.setMouseCallback('mask_', draw_roi)
    global keep_mask
    global drop_mask
    global pts
    mask = np.zeros_like(img)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break
        if key == ord("s"):
            cv2.fillPoly(mask, [np.array(pts)], (255, 255, 255))
            cv2.imshow("mask", mask)
            pts = []
            if opt == 1:
                keep_mask = mask[:, :, 0].copy()
            else:
                drop_mask = mask[:, :, 0].copy()


def showMask(opt):
    if opt == 1:
        cv2.imshow("keepmask", keep_mask)
    else:
        cv2.imshow("dropmask", drop_mask)


def run_seam_carve():
    height = int(height_input.get())
    width = int(width_input.get())
    energy = energy_input.get()
    order = order_input.get()
    remove = option2.get()
    global img_output
    if current == "resize":
        if isKeepMask.get():
            img_output = seam_carving.resize(
                img_input, (width, height), energy, order, keep_mask=keep_mask)
        else:
            img_output = seam_carving.resize(
                img_input, (width, height), energy, order, keep_mask=None)
    else:
        keep_mask_ = None
        if remove == "width":
            if isKeepMask.get():
                img_output, keep_mask_ = seam_carving.remove_object_width(
                    img_input, drop_mask, keep_mask)
            else:
                img_output, _ = seam_carving.remove_object_width(
                    img_input, drop_mask)
        else:
            if isKeepMask.get():
                img_output, keep_mask_ = seam_carving.remove_object_height(
                    img_input, drop_mask, keep_mask)
            else:
                img_output, _ = seam_carving.remove_object_height(
                    img_input, drop_mask)
        if isKeepSize.get():
            img_output = seam_carving.resize(
                img_output, (src_w, src_h), energy, order, keep_mask_)
    cv2.imshow("output", img_output)


# def callbackFunc(event):
#     global current
#     selected_value = event.widget.get()
#     if selected_value == "Resize":
#         current = "resize"
#         width_input.place(x=1420, y=105)
#         height_input.place(x=1420, y=75)
#         energy_input.place(x=1420, y=135)
#         order_input.place(x=1420, y=165)
#         is_keep_mask.place(x=1420, y=195)
#         keep_mask_btn.place(x=1420, y=225)
#         show_mask_btn1.place(x=1420, y=255)
#         drop_mask_btn.place(x=-50, y=-50)
#         show_mask_btn2.place(x=-50, y=-50)
#         is_drop_mask.place(x=-50, y=-50)
#         drop_mask_btn.place(x=-50, y=-50)
#         show_mask_btn2.place(x=-50, y=-50)
#         is_keep_size.place(x=-50, y=-50)
#         option2.place(x=-50, y=-50)
#     else:
#         current = "remove"
#         is_keep_mask.place(x=1420, y=75)
#         keep_mask_btn.place(x=1420, y=105)
#         show_mask_btn1.place(x=1420, y=135)
#         is_drop_mask.place(x=1420, y=165)
#         drop_mask_btn.place(x=1420, y=195)
#         show_mask_btn2.place(x=1420, y=225)
#         is_keep_size.place(x=1420, y=255)
#         option2.place(x=1420, y=285)
#         width_input.place(x=-50, y=-50)
#         height_input.place(x=-50, y=-50)
#         order_input.place(x=-50, y=-50)
#         energy_input.place(x=-50, y=-50)


def exportImg():
    defaultextension = '.png'
    filename = fd.asksaveasfile(mode='w', defaultextension=defaultextension)
    if not filename:
        return
    cv2.imwrite(filename.name, img_output)


window = Tk()
window.geometry('1600x800')
window.resizable(0, 0)
my_label = Button(window)

import_button = Button(window, text='Import image', bg='black', fg='white')
import_button.config(command=lambda: importImg(my_label))
import_button.config(height=2, width=15)
import_button.place(x=25, y=750)

import_button = Button(window, text='Export image', bg='black', fg='white')
import_button.config(command=lambda: exportImg())
import_button.config(height=2, width=15)
import_button.place(x=150, y=750)


# option = ttk.Combobox(window, height=50, width=15)
# option['values'] = ('Remove Object')
# option.place(x=1420, y=45)
# option.current(0)
# option.bind("<<ComboboxSelected>>", callbackFunc)

height_input = ttk.Entry(window, width=15)
width_input = ttk.Entry(window, width=15)
width_input.place(x=1420, y=105)
height_input.place(x=1420, y=75)

done_button = Button(window, text='Done', bg='black', fg='white')
done_button.config(command=lambda: run_seam_carve())
done_button.config(height=2, width=15)
done_button.place(x=1420, y=750)

energy_input = ttk.Combobox(window, width=15)
energy_input['values'] = ('backward',
                          'forward')
energy_input.current(0)
energy_input.place(x=1420, y=135)

order_input = ttk.Combobox(window, width=15)
order_input['values'] = ('width-first',
                         'height-first',
                         'optimal')
order_input.current(0)
order_input.place(x=1420, y=165)

isKeepMask = BooleanVar()
is_keep_mask = Checkbutton(
    window, text='Use keep mask', variable=isKeepMask)
is_keep_mask.place(x=1420, y=195)
keep_mask_btn = ttk.Button(window, text='keep mask',
                           command=lambda: drawMask(1))
keep_mask_btn.place(x=1420, y=225)
show_mask_btn1 = ttk.Button(
    window, text='show keep mask', command=lambda: showMask(1))
show_mask_btn1.place(x=1420, y=255)


isDropMask = BooleanVar()
is_drop_mask = Checkbutton(
    window, text='Use drop mask', variable=isDropMask)
drop_mask_btn = ttk.Button(window, text='drop mask',
                           command=lambda: drawMask(0))
show_mask_btn2 = ttk.Button(
    window, text='show drop mask', command=lambda: showMask(0))
isKeepSize = BooleanVar()
is_keep_size = Checkbutton(
    window, text='Keep original size', variable=isKeepSize)

option2 = ttk.Combobox(window, height=50, width=15)
option2['values'] = ('width',
                     'height')
option2.current(0)

current = "remove"
is_keep_mask.place(x=1420, y=75)
keep_mask_btn.place(x=1420, y=105)
show_mask_btn1.place(x=1420, y=135)
is_drop_mask.place(x=1420, y=165)
drop_mask_btn.place(x=1420, y=195)
show_mask_btn2.place(x=1420, y=225)
is_keep_size.place(x=1420, y=255)
option2.place(x=1420, y=285)
width_input.place(x=-50, y=-50)
height_input.place(x=-50, y=-50)
order_input.place(x=-50, y=-50)
energy_input.place(x=-50, y=-50)

window.mainloop()
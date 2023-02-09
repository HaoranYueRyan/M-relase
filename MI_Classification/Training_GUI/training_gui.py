from tkinter import *
from omero_images import OmeroImages
from PIL import Image, ImageTk
import numpy as np

BACKGROUND_COLOR = "#375362"
count = 0


class TrainingScreen:
    def __init__(self, imgs: OmeroImages):
        self.dict_all = {
            'data':[],
            'target':[],
        }
        self.nuclei = imgs
        self.window = Tk()
        self.window.title('Mitotic Index Training')
        self.window.config(padx=20, pady=20, bg=BACKGROUND_COLOR)
        self.canvas = Canvas(width=500, height=500)
        self.canvas.config(bg='grey', highlightthickness=0)
        self.canvas.grid(row=1, column=1, columnspan=2, pady=20)
        self.generate_image()
        self.image_container = self.canvas.create_image(250, 250, image=self.photo)
        self.next_button = Button(
            # text="Show next Nuclei",
            text="Show next Cell",
            font=("arial", 20, "italic"),
            fg=BACKGROUND_COLOR,
            bg='white',
            highlightthickness=0,
            command=self.update_image,
        )
        self.next_button.grid(row=0, column=1, columnspan=2)

        self.back_button = Button(
            text="Step Back",
            font=("arial", 20, "italic"),
            fg=BACKGROUND_COLOR,
            bg='white',
            highlightthickness=0,
            command=self.step_back,
        )
        self.back_button.grid(row=4, column=1, columnspan=2)

        self.mitosis = Button(
            text="Mitosis",
            font=("arial", 30, "italic"),
            fg=BACKGROUND_COLOR,
            bg='white',
            highlightthickness=0,
            command=self.classify_mitosis,
        )
        self.mitosis.grid(row=3, column=0)

        self.interphase = Button(
            text="Interphase",
            font=("arial", 30, "italic"),
            fg=BACKGROUND_COLOR,
            bg='white',
            highlightthickness=0,
            command=self.classify_interphase,
        )
        self.interphase.grid(row=3, column=3)

        self.counter = Label(text=f'image {count} of {len(self.nuclei.nuclei_list)}')
        self.counter.grid(row=1, column=3)


        self.window.mainloop()

    def update_image(self):
        global count

        count += 1
        self.generate_image()
        self.canvas.itemconfig(self.image_container, image=self.photo)
        self.counter.config(text=f'image {count} of {len(self.nuclei.nuclei_list)}')



    def step_back(self):
        global count
        count -= 1
        self.generate_image()
        self.counter.config(text=str(count))
        self.canvas.itemconfig(self.image_container, image=self.photo)
        for k, v in self.dict_all.items():
            v.pop()


    def generate_image(self):
        if count < len(self.nuclei.nuclei_list):
            img_16 =self.nuclei.nuclei_list[count]
            img_8 = (img_16 / img_16.max()) * 255
            img_8 = np.uint8(img_8)
            image = Image.fromarray(img_8)
            resize_image = image.resize((450, 450))
            self.photo = ImageTk.PhotoImage(resize_image)
        else:
            np.save(f'../CNN_Training/TrainingData/{self.nuclei.image_ID}.npy',self.dict_all)

    def classify_mitosis(self):
        if count < len(self.nuclei.nuclei_list):
            self.dict_all['target'].append(1)
            self.dict_all['data'].append(self.nuclei.nuclei_list[count])
            self.update_image()
        else:
            self.counter.config(text='data saved')


    def classify_interphase(self):
        if count < len(self.nuclei.nuclei_list):
            self.dict_all['target'].append(0)
            self.dict_all['data'].append(self.nuclei.nuclei_list[count])
            self.update_image()
        else:
            self.counter.config(text='data saved')


from omero_images import OmeroImages
from training_gui import TrainingScreen, count
import matplotlib.pyplot as plt

plate_id = 9674
nuclei_per_img = 100

imgs = OmeroImages(plate_id, nuclei_per_img)

screen = TrainingScreen(imgs).astype(dtype='Int64')
# print(count)




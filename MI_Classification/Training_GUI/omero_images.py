import matplotlib.pyplot as plt
import numpy as np
from omero.gateway import BlitzGateway
from cellpose import models
from skimage import (
    io, exposure, feature, filters, measure, segmentation, color, morphology
)
import random
from config import username, password
from aggregator import *
import pandas as pd


class OmeroImages:

    def __init__(self, well_id,  sample_number):
        self.image_ID = well_id
        # self.timepoints = timepoints
        self.sample_number = sample_number
        self.aggregate_imgs_DPAI=self._aggregate_imgs(well_id,0)
        self.aggregate_imgs_tubulins = self._aggregate_imgs(well_id, 3)
        self.image_list = self.fetch_images(well_id)
        self.width = 20
        self.nuclei_list = self.get_nuclei(self.image_list)

    def fetch_images(self,well_id):
        """
        Fetch images from the OMERO server.
        :param well_id: ID of the well to fetch images from
        :return: List of images fetched from the OMERO server
        """

        image_list=[]
        conn = BlitzGateway(username, password, host='ome2.hpc.sussex.ac.uk')
        conn.connect()
        well = conn.getObject("Well", well_id)
        # defined the DAPI(0) and tubulin(3) channel
        # aggregate_imgs_DPAI=self._aggregate_imgs(well,0)
        # aggregate_imgs_tubulins = self._aggregate_imgs(well, 3)
        for i, img in enumerate(well.listChildren()):
            image_object = well.getImage(i)
            pixels = image_object.getPrimaryPixels()
            corr_DPAI_image =self._gen_example(pixels.getPlane(0, 0, 0),self.aggregate_imgs_DPAI)
            corr_tubulins_image = self._gen_example(pixels.getPlane(0, 3, 0), self.aggregate_imgs_tubulins)
            empty_channel=np.zeros((corr_tubulins_image.shape[0],corr_tubulins_image.shape[1]))
            comb_image= np.dstack([empty_channel,corr_tubulins_image,corr_DPAI_image]).astype('float32')
            image_list.append(self._scale_img(comb_image))
        conn.close()
        return image_list

    def _aggregate_imgs(self,well_id, channel_number):
        """
        Aggregate images in well for specified channel and generate correction mask using the Aggregator Module
        :param channel_number: Channel number to aggregate images for
        :return: flatfield correction mask for given channel
        """
        conn = BlitzGateway(username, password, host='ome2.hpc.sussex.ac.uk')
        conn.connect()
        well = conn.getObject("Well", well_id)

        agg = ImageAggregator(60)
        for i, img in enumerate(well.listChildren()):
            image = well.getImage(i)
            image_array = self._generate_image(image, channel_number)
            agg.add_image(image_array)
        blurred_agg_img = agg.get_gaussian_image(30)

        conn.close()
        return blurred_agg_img / blurred_agg_img.mean()

    def _gen_example(self,example_img, mask):

        corr_img = example_img / mask
        bg_corr_img = corr_img - np.median(corr_img)
        bg_corr_img[np.where(bg_corr_img <= 100)] = 100
        corr_scaled = self._scale_img(bg_corr_img)
        # order all images for plotting
        return corr_scaled

    def _scale_img(self,img: np.array, percentile: tuple[float, float] = (1, 99)) -> np.array:
        """Increase contrast by scaling image to exclude lowest and highest intensities"""
        percentiles = np.percentile(img, (percentile[0], percentile[1]))
        return exposure.rescale_intensity(img, in_range=tuple(percentiles))

    def _generate_image(self,image: 'Omero image object', channel: int) -> np.ndarray:
        """
        Turn Omero Image Object from Well into numpy nd-array that is returned

        :param channel: channel number
        :return: numpy.ndarray
        """
        pixels = image.getPrimaryPixels()
        return pixels.getPlane(0, channel, 0)  # using channel number

    def get_nuclei(self, image_list):
        nuclei_list = []
        for img in image_list:
            # n_model = models.CellposeModel(gpu=False, model_type='cyto')
            cyto2_tubulin_model_path = '/Users/haoranyue/PycharmProjects/Omero_Screen/Mitotic-Release/data/CellPose_models/RPE-1_Tub_Hoechst'
            n_model = models.CellposeModel(gpu=False, pretrained_model=cyto2_tubulin_model_path)
            n_channels = [[2, 3]]
            n_masks, n_flows, n_styles = n_model.eval(img, diameter=30.8, channels=n_channels)
            df_props=pd.DataFrame(measure.regionprops_table(n_masks, properties=('label','centroid',)))
            for label in random.sample(df_props['label'].tolist(), self.sample_number):

                # centroid = region.centroid
                i = df_props.loc[df_props['label']==label,'centroid-0'].item()

                j = df_props.loc[df_props['label']==label,'centroid-1'].item()

                imin = int(round(max(0, i-self.width)))
                imax = int(round(min(n_masks.shape[0], i+self.width+1)))
                jmin = int(round(max(0, j-self.width)))
                jmax = int(round(min(n_masks.shape[1], j+self.width+1)))
                img[:, :, 0] = n_masks
                img[:, :, 0] = (img[:, :, 0]==label) * np.ones((img[:, :, 0].shape[0], img[:, :, 0].shape[1]))
                box=img[imin:imax, jmin:jmax].copy()
                nuclei_list.append(box)
                del box

        return nuclei_list

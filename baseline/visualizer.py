import os
from PIL import Image
import numpy as np
import pytz
from datetime import datetime
import config
from monai.visualize import plot_2d_or_3d_image
from monai.data import decollate_batch

class Visualizer():

    def __init__(self, val_dataloader, inferer, num_samples, writer, post_trans):
        self.inferer = inferer
        self.dataloader = val_dataloader
        self.num_samples = num_samples
        self.dir = self.create_dir()
        self.writer = writer
        self.post_trans = post_trans

        self.images_original, self.images_array = self.get_data()



    def get_data(self):
        images_original = []
        images_array = []
        for i in range(self.num_samples):
            data = next(iter(self.dataloader))
            image_dl = data['image']
            label_dl = data['label']

            if i == 0:
                plot_2d_or_3d_image(image_dl, 0, self.writer, index=0, tag="image")
                plot_2d_or_3d_image(label_dl, 0, self.writer, index=0, tag="label")

            # image as .jpg
            image_array = np.squeeze(image_dl.numpy())
            image_image = Image.fromarray(np.uint8(image_array * 255))
            image_image.save(self.dir + str(i+1) + '/' + 'image.jpg')

            # overlay as .jpg
            label_array = np.squeeze(label_dl.numpy())
            overlay = self.compute_overlay(label_array=label_array, image_array=image_array)
            overlay.save(self.dir + str(i+1) + '/' + 'label.jpg')

            images_original.append(image_dl)
            images_array.append(image_array)

        #return images, labels, overlays, images_original
        return images_original, images_array


    def compute_overlay(self, label_array, image_array):
        eps = 1e-10
        mask = np.where(label_array > eps)
        overlay = image_array.copy()
        overlay[mask] = label_array[mask]
        overlay = Image.fromarray(np.uint8(overlay * 255))
        return overlay


    def create_dir(self):
        tz = pytz.timezone('Europe/Berlin')
        now = str(datetime.now(tz))
        folder_name = config.user + "_" + now[:10] + "_" + now[11:19].replace(':', '-')
        dir = 'visualizations/' + folder_name + '/sample'
        for i in range(self.num_samples):
            sample_dir = dir + str(i+1) + '/'
            os.makedirs(sample_dir)
        return dir

   
    def visualize(self, model, epoch):

        for i in range(self.num_samples):
            # prediction overlay
            pred = self.inferer(inputs=self.images_original[i].to(config.device), network=model)
            # pred = model(self.images_original[i].to(config.device))
            pred_array = np.squeeze(pred.cpu().detach().numpy())
            pred_array = 1 * np.greater_equal(pred_array, 0.5)
            pred_image = Image.fromarray(np.uint8(pred_array * 255))

            sample_dir = self.dir + str(i+1) + '/'
            filename = 'epoch' + str(epoch) + '.jpg'
            pred_image.save(sample_dir + filename)

            if i == 0:
                val_outputs = [self.post_trans(i) for i in decollate_batch(pred)]
                plot_2d_or_3d_image(val_outputs, epoch, self.writer, index=0, tag="output")


       


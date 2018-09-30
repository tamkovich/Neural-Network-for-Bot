import numpy
import matplotlib.pyplot as plt

from PIL import Image as PIL_Image


class Image:
    def resize_image(self, input_image_path,
                     size):
        original_image = PIL_Image.open(input_image_path)

        resized_image = original_image.resize(size)
        return resized_image

    def prepare_image(self, filename):
        data = self.resize_image(
            input_image_path=filename,
            size=(28, 28)
        )

        data = numpy.asarray(data)
        data = data.ravel()[::3]

        img_data = 255.0 - data.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01

        return img_data

    # just for neural-network method
    @staticmethod
    def save_image(network, label, update_enen_if_exists=False):
        targets = numpy.zeros(network.onodes) + 0.01
        targets[label] = 0.99

        image_data = network.backquery(targets)
        plt.imsave(f'img_num_10/{label}.png', image_data.reshape(28, 28), cmap='Greys')

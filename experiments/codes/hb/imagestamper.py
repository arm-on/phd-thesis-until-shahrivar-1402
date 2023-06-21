import numpy as np


def abstractmethod(args):
    pass


class ImageStamper:
    def __init__(self, src, patch):
        """
        src: source image (np-array)
        patch: trigger (patch) image (np-array)
        """
        self.src = src.copy()
        self.patch = patch.copy()

    @abstractmethod
    def stamp(self):
        pass


class Stamper2D(ImageStamper):
    def __init__(self, src, patch):
        """
        note: source and patch should both be 2D images (channel-last format)
        """
        super().__init__(src, patch)

    def stamp(self, x_min, y_min):
        """
        x_min: a number
        y_min: a number
        """
        patch_width, patch_height = self.patch.shape[0], self.patch.shape[1]
        stamped_img = self.src.copy()
        stamped_img[x_min:x_min + patch_width, y_min:y_min + patch_height] = self.patch
        return stamped_img


class Stamper3D(ImageStamper):
    def __init__(self, src, patch):
        """
        note: source and patch should both be 3D images (channel-last format)
        """
        super().__init__(src, patch)

    def stamp(self, x_min, y_min):
        """
        x_min: a number
        y_min: a number
        """
        patch_width, patch_height = self.patch.shape[0], self.patch.shape[1]
        stamped_img = self.src.copy()
        stamped_img[x_min:x_min + patch_width, y_min:y_min + patch_height, :] = self.patch
        return stamped_img


def random_bulk_stamper(images, patch):
    """
    patch: the image used as the patch (channels-last format)
    """
    patch_width, patch_height = patch.shape[0], patch.shape[1]
    image_width, image_height = images[0].shape[0], images[0].shape[1]
    num_dims = 3 if len(images[0].shape) == 3 else 2
    stamper_obj = Stamper3D if num_dims == 3 else Stamper2D
    stamped_images = []
    for img in images:
        stamper = stamper_obj(img, patch)
        x_min = np.random.randint(low=0, high=image_width-patch_width)
        y_min = np.random.randint(low=0, high=image_height-patch_height)
        stamped_img = stamper.stamp(x_min, y_min)
        stamped_images.append(stamped_img)
    stamped_images = np.array(stamped_images)
    return stamped_images

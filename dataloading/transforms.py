import torchvision.transforms as Transforms
# import numpy as np
# import dlib
import torch
import logging

#
# class PILTransforms:
#     @staticmethod
#     def crop_face():
#         def crop(image):
#             image_ary = np.array(image)
#
#             # detect and crop face
#             detector = dlib.get_frontal_face_detector()
#             det = detector(image_ary)
#
#             if len(det) > 0:
#                 d = det[0]
#
#                 box = (d.left(), d.top(), d.right(), d.bottom())
#
#                 image = image.crop(box)
#
#             return image
#
#         return Transforms.Lambda(crop)


class SlidingWindowTransformClass:
    def __init__(self, size, step=None, threshold=None):
        self.size = size
        self.step = size if step is None else step
        self.threshold = threshold
        self.comp = lambda _, __: True

        sliding_window = None
        if threshold is None:
            sliding_window = self.get_sliding_window_simple()
        else:
            sliding_window = self.get_sliding_window_with_threshold()

        self.transform = Transforms.Lambda(sliding_window)

    def get_sliding_window_with_threshold(self):
        assert self.threshold is not None, "Threshold can't be none"
        self.comp = lambda mean, min: mean < min
        if type(self.threshold) is tuple:
            self.comp = lambda mean, value: mean < value[0] or mean > value[1]

        def sliding_window(x):
            # unfold dimension to make our rolling window
            windows = x.unfold(1, self.size, self.step).unfold(2, self.size, self.step)
            patches = []
            for i in range(len(windows[0, :, 0, 0, 0])):
                for n in range(len(windows[0, 0, :, 0, 0])):
                    if not self.comp(torch.mean(windows[0, i, n, :, :]), self.threshold):
                        patches.append(windows[0, i, n, :, :])
                    else:
                        logging.debug('discarded patch [{}|{}]'.format(i, n))
            del windows

            if len(patches) > 0:
                patches = torch.stack(patches)
                return patches
            else:
                logging.info('no patches found with threshold {}'.format(self.threshold))
                self.threshold *= 1.01
                if self.threshold > 1:
                    logging.error('no patches found with threshold {}'.format(self.threshold))
                    return torch.Tensor()
                return sliding_window(x, self.threshold)
        return sliding_window

    def get_sliding_window_simple(self):
        def sliding_window(x):
            # unfold dimension to make our rolling window
            windows = x.unfold(1, self.size, self.step).unfold(2, self.size, self.step)
            patches = []
            for i in range(len(windows[0, :, 0, 0, 0])):
                for n in range(len(windows[0, 0, :, 0, 0])):
                    patches.append(windows[0, i, n, :, :])
            del windows

            patches = torch.stack(patches)
            return patches
        return sliding_window

    def __call__(self, *args, **kwargs):
        return self.transform(*args)


class CombiningWindowTransformClass:
    def __init__(self, size, step):
        # self.width, self.height = size if len(size) >= 2 else size, size
        self.width, self.height = size
        self.step = step

        combining_window = self.get_combining_window()

        self.transform = Transforms.Lambda(combining_window)

    def get_combining_window(self):
        # def combining_window(images):
        #     fold = torch.nn.Fold((self.height, self.width), kernel_size=(self.step, self.step))
        #     img = fold(images)
        #     return img

        def combining_window(images):
            reconstructed = torch.Tensor(self.height, self.width)
            x = 0
            y = 0
            width = self.width // self.step
            height = self.height // self.step
            y_start = y * self.step
            y_end = (y + 1) * self.step
            for p in images:
                x_start = x * self.step
                x_end = (x + 1) * self.step

                if x_end > self.width:
                    p = p[:self.step, :self.width - x_start]
                # if not x_end > self.width:
                #     reconstructed[y_start:y_end, x_start:x_end] = p
                reconstructed[y_start:y_end, x_start:x_end] = p

                x += 1
                if x >= width:
                    if y >= height:
                        break
                    x = 0
                    y += 1
                    y_start = y * self.step
                    y_end = (y + 1) * self.step
            return reconstructed
        return combining_window

    def __call__(self, *args, **kwargs):
        return self.transform(*args)


class DualTransforms:
    def __init__(self, transforms, shared_transforms=None, shared_transforms_post=None):
        if isinstance(transforms, (list, tuple)):
            transforms = Transforms.Compose(transforms)
        if isinstance(shared_transforms, (list, tuple)):
            shared_transforms = Transforms.Compose(shared_transforms)
        if isinstance(shared_transforms_post, (list, tuple)):
            shared_transforms_post = Transforms.Compose(shared_transforms_post)
        self.transforms = transforms
        self.shared_transfroms = shared_transforms
        self.shared_transforms_post = shared_transforms_post

        self.transform = Transforms.Lambda(self.get_dual_transforms())

    def get_dual_transforms(self):
        def dual_transforms(x):
            if self.shared_transfroms is not None:
                x, y = self.shared_transfroms(x), self.shared_transfroms(x)
            y = self.transforms(x)
            if self.shared_transforms_post is not None:
                x, y = self.shared_transforms_post(x), self.shared_transforms_post(y)
            return x, y
        return dual_transforms

    def __call__(self, *args, **kwargs):
        return self.transform(*args)


class NTransforms:
    def __init__(self, n, shared_transforms=None, transforms=None, shared_transforms_post=None):
        assert 0 < n == len(transforms), "n must be greater than 0 and len(transforms) must be equal to n"
        self.transforms = [Transforms.Compose(transform) for transform in transforms]

        if isinstance(shared_transforms, (list, tuple)):
            shared_transforms = Transforms.Compose(shared_transforms)
        if isinstance(shared_transforms_post, (list, tuple)):
            shared_transforms_post = Transforms.Compose(shared_transforms_post)

        self.shared_transfroms = shared_transforms
        self.shared_transforms_post = shared_transforms_post

        self.transform = Transforms.Lambda(self.get_n_transforms())

    def get_n_transforms(self):
        def n_transforms(x):
            if self.shared_transfroms is not None:
                x = self.shared_transfroms(x)
            nres = [transform(x) for transform in self.transforms]
            if self.shared_transforms_post is not None:
                nres = [self.shared_transforms_post(res) for res in nres]
            return nres
        return n_transforms

    def __call__(self, *args, **kwargs):
        return self.transform(*args)

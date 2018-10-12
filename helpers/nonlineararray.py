import torch
import numpy as np
import time


class FmcwNonlinearArray:
    """ FMCW nonlinear antenna array """
    def __init__(self):
        # fmcw array coefficients for given region of interest
        self.HOR_COEF_MATRIX = None
        self.VER_COEF_MATRIX = None

    def compute_heatmap(self, frame, array_type):
        if array_type == "HOR":
            response = np.dot(self.HOR_COEF_MATRIX, frame)
            heatmap = response.reshape(self.length_x, self.length_z)
        elif array_type == "VER":
            response = np.dot(self.VER_COEF_MATRIX, frame)
            heatmap = response.reshape(self.length_y, self.length_z)
        else:
            raise RuntimeError("Unknown type of antenna array.")
        return heatmap

    def load_coefficients(self, coef_filename, length_x, length_y, length_z):
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z
        print("Loading nonlinear array coefficients...")
        start = time.time()
        coefficients = np.load(coef_filename)
        self.HOR_COEF_MATRIX = coefficients["hor"]
        self.VER_COEF_MATRIX = coefficients["ver"]
        print("Coefficients load done in {:.2f}s.".format(time.time() - start))


class FmcwNonlinearArrayGPU:
    """ FMCW nonlinear antenna array """
    def __init__(self):
        # fmcw array coefficients for given region of interest
        self.coef_real = [None] * 2
        self.coef_imag = [None] * 2

    def compute_heatmap(self, frame, array_type):
        frame_gpu_real = torch.from_numpy(frame).cuda()

        if array_type == "HOR":
            idx = 0
        elif array_type == "VER":
            idx = 1
        else:
            raise RuntimeError("Unknown type of antenna array.")

        real = torch.mm(self.coef_real[idx], frame_gpu_real.view(-1, 1))
        imag = torch.mm(self.coef_imag[idx], frame_gpu_real.view(-1, 1))

        return torch.stack([real, imag])

    def load_coefficients(self, coef_filename):
        print("Loading nonlinear array coefficients...")

        coef = [None] * 2

        start = time.time()
        coefficients = np.load(coef_filename)
        coef[0] = coefficients["hor"]
        coef[1] = coefficients["ver"]
        print("Coefficients load done in {:.2f}s.".format(time.time() - start))

        for idx in range(2):
            self.coef_real[idx] = torch.from_numpy(np.real(coef[idx])).cuda()
            self.coef_imag[idx] = torch.from_numpy(np.imag(coef[idx])).cuda()

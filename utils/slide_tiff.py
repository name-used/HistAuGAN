# @Time    : 2022.11.21
# @Author  : kawa Yeung
# @Licence : bio-totem
import os

import multiresolutionimageinterface as mir
import tifffile
import tiffslide
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class Slide(tiffslide.TiffSlide):
    def __init__(self, svs_file, level=2):
        """
        open svs file with open-slide
        :param svs_file: svs file, absolute path
        :return: slide
        """
        super().__init__(svs_file)
        self._filepath = svs_file
        self._basename = os.path.basename(svs_file).split('.')[0]
        self.slide = tiffslide.TiffSlide(svs_file)
        self._level = level

    def get_basename(self):
        """
        return svs file basename, not contain file suffix
        :return:
        """

        return self._basename

    def get_filepath(self):
        """
        get absolute svs file
        :return:
        """

        return self._filepath

    def get_level(self):
        """
        return level
        :return:
        """

        return self._level

    def get_level_downsample(self, level=2):
        """
        get the expected level downsample, default level two
        :param level: level, default 2
        :return: the level downsample
        """

        return self.slide.level_downsamples[level]

    def get_level_dimension(self, level=2):
        """
        get the expected level dimension, default level two
        :param level: level, default 0
        :return:
        """

        return self.slide.level_dimensions[level]

    def get_thumb(self, level=2):
        """
        get thumb image
        :return:
        """

        level_dimension = self.get_level_dimension(level)
        # print(level_dimension)
        tile = self.slide.get_thumbnail(level_dimension)

        return tile

    def svs_to_png(self, save_dir):
        """
        convert svs to png
        :return:
        """
        self.get_thumb().save(save_dir)

    def expand_img(self, im, size, value=(0, 0, 0)):
        """
        expand the image
        :param im: the image want to expand
        :param size: tuple, the size of expand
        :param value: tuple, the pixel value at the expand region
        :return: the expanded image
        """

        im_new = Image.new("RGB", size, value)
        im_new.paste(im, (0, 0))

        return im_new

    def get_mpp(self):
        """
        get the value of mpp
        :return: 0.00025
        """
        properties = self.properties
        properties['tiffslide.mpp-x']
        return np.float(properties['tiffslide.mpp-x']) / 1000


class Writer(mir.MultiResolutionImage):
    def __init__(self, output_path: str, tile_size: int, dimensions: tuple, spacing: float, *args, **kwargs):
        self.output_path = output_path
        self.tile_size = tile_size
        self.W, self.H = dimensions
        # 要求横纵分辨率一致
        self.spacing = spacing

    def __enter__(self):
        self._writer = mir.MultiResolutionImageWriter()
        self._writer.openFile(self.output_path)
        self._writer.setTileSize(self.tile_size)
        self._writer.setCompression(mir.LZW)
        self._writer.setDataType(mir.UChar)
        self._writer.setInterpolation(mir.NearestNeighbor)
        self._writer.setColorType(mir.Monochrome)
        self._writer.writeImageInformation(self.W, self.H)
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(self.spacing)
        pixel_size_vec.push_back(self.spacing)
        self._writer.setSpacing(pixel_size_vec)
        return self

    def write(self, tile: np.ndarray, x: int, y: int):
        assert tile.shape[0] == tile.shape[1] == self.tile_size, '要求写入数与维度数对齐'
        self._writer.writeBaseImagePartToLocation(tile.flatten().astype('uint8'), x=int(x), y=int(y))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # import traceback
        # traceback.print_exc()
        self._writer.finishImage()
        # if exc_type or exc_val or exc_tb:
        # return False

import numpy as np
from PIL import Image
from os.path import *
import re
import cv2

import OpenEXR
import Imath

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2 ** 15) / 64.0
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2 ** 15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


### 代码误删重新编写readExr部分，2025年8月20日 
def readExr(filename):
    """
    输出图像文件:h*w*3 图像文件h*w*2
    """
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    # 获取所有通道的名称
    channels = header["channels"].keys()
    # 判断是图像 (3通道) 还是光流 (2通道)
    if len(channels) >= 3:
        # 读取 RGB 通道
        if all((c in channels for c in ('R', 'G', 'B'))):
            pixel_type = header["channels"]["R"].type
            channel_data = exr_file.channels(["R", "G", "B"], pixel_type)
            r = np.frombuffer((channel_data[0]), dtype=(np.float32)).reshape(height, width)
            g = np.frombuffer((channel_data[1]), dtype=(np.float32)).reshape(height, width)
            b = np.frombuffer((channel_data[2]), dtype=(np.float32)).reshape(height, width)
            # 将通道堆叠成 HxWx3 的图像数组
            return np.stack([r, g, b], axis=(-1))
    if len(channels) >= 2:
        # 读取 RG 通道 (通常用于光流)
        if all((c in channels for c in ('R', 'G'))):
            pixel_type = header["channels"]["R"].type
            channel_data = exr_file.channels(["R", "G"], pixel_type)
            r = np.frombuffer((channel_data[0]), dtype=(np.float32)).reshape(height, width)
            g = np.frombuffer((channel_data[1]), dtype=(np.float32)).reshape(height, width)
            # 将通道堆叠成 HxWx2 的光流数组
            return np.stack([r, g], axis=(-1))
    if len(channels) == 1:
        # 读取R为深度通道
         if all((c in channels for c in ('R'))):
            pixel_type = header["channels"]["R"].type
            channel_data = exr_file.channels(["R"], pixel_type)
            r = np.frombuffer((channel_data[0]), dtype=(np.float32)).reshape(height, width)
            # 将通道堆叠成 HxWx1 的光流数组
            return np.stack([r], axis=(-1))
    print(f"{filename} 中的通道不符合标准 (RGB 或 RG)，无法加载。")
    return


### 代码误删重新编写read_exr部分，2025年8月20日 
def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]

    if  ext == '.exr':
        return readExr(file_name)
    elif ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []

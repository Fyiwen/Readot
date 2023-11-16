# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

import os
import sys
import collections
import numpy as np
import struct


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"): # 用于从二进制文件中读取和解包下一组字节的函数
    """Read and unpack the next bytes from a binary file.
    :param fid: 要处理的文件对象
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.要读取或者说是解包的字节数。它是2、4、8等的组合，例如2、6、16、30等
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.格式字符的序列，用于指定读取数据的类型。它是以下字符之一的列表：c、e、f、d、h、H、i、I、l、L、q、Q。
    :param endian_character: Any of {@, =, <, >, !}表示字节序的字符，用于指定如何解析二进制数据。它可以是@、=、<、>、!中的任意一个
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)  # 从文件中读取num_bytes个字节赋予data变量
    return struct.unpack(endian_character + format_char_sequence, data) # 返回据给定的格式字符序列和字节序对二进制数据进行解包后的结果


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file): # 给定了cameras.bin文件所在的路径
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid: # 打开cameras.bin文件
        num_cameras = read_next_bytes(fid, 8, "Q")[0] # 从fid文件中读取8个字节的内容后进行解析，结果为28（因为colmap从28张图片中处理出了28个相机姿态）
        for camera_line_index in range(num_cameras): # 遍历这28个
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ") # 从fid文件中读取24个字节，按照iiqq格式解包后结果为（1，0，3000，4000）
            camera_id = camera_properties[0] # 1
            model_id = camera_properties[1] # 0
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name # SIMPLE_PINHOLE.我用colmap做特征抽取的时候确实选了这个参数
            width = camera_properties[2] # 3000
            height = camera_properties[3] # 4000
            num_params = CAMERA_MODEL_IDS[model_id].num_params # 3
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params) #从文件中读取后解析为(3158.825714548802, 1500.0, 2000.0)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))  # 将这些参数做成元组。以字典的形式存在cameras变量里
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file): # 输入images.bin所在路径
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid: # 读取images.bin文件
        num_reg_images = read_next_bytes(fid, 8, "Q")[0] # 从文件中读取8个字节解析后得到的结果为28，也就是说有28个图像信息
        for image_index in range(num_reg_images): # 遍历这28个
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi") # 读取到的结果解析为(1, 0.9611202077547979, -0.027466382035340572, 0.27390479974022763, 0.021672673635910394, -3.137882246799287, -1.3976252965098346, 2.5225788440911034, 1)，一共9个
            image_id = binary_image_properties[0] # 1
            qvec = np.array(binary_image_properties[1:5]) #[ 0.96112021 -0.02746638  0.2739048   0.02167267]
            tvec = np.array(binary_image_properties[5:8])  # [-3.13788225 -1.3976253   2.52257884]
            camera_id = binary_image_properties[8] #1
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0] # 又从fid里面尝试读了一个字节，看看现在已经读取到了这个文件的哪个位置
            while current_char != b"\x00": # 如果现在读到的位置不是这个，就先把现在读到的内容编码成utf_8形式作为图像名称的一个部分，再继续往下读一个字节，知道读到这个位置为止  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8") # 几轮下来后结果为1.jpg
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0] # 又继续读读到的结果是3971
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D) # 读到的结果是个元组有11913个
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), # 从0开始每隔3个取一个元素
                                   tuple(map(float, x_y_id_s[1::3]))]) # 最终形状[3971,2]
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3]))) # 一共3971个
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids) # 把上面得到的图片信息以字典形式存在images变量对应索引位置里
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid: # 读取这个文件
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file): # # 输入point3D.bin文件的路径
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid: # 读取point3D.bin文件
        num_points = read_next_bytes(fid, 8, "Q")[0] # 读取文件中前8个字节，解析得到点的总数。4404
        for point_line_index in range(num_points): # 遍历每一个点
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd") # 读取后解析，得到关于这个点的所有参数。有8个(1, 0.9982301099252779, -1.7666323544568636, 4.624652719959426, 171, 163, 144, 1.9537206078743516)
            point3D_id = binary_point_line_properties[0] # 1
            xyz = np.array(binary_point_line_properties[1:4]) # [ 0.99823011 -1.76663235  4.62465272]
            rgb = np.array(binary_point_line_properties[4:7]) # [171 163 144]
            error = np.array(binary_point_line_properties[7]) # array(1.95372061)
            track_length = read_next_bytes( # 跟踪长度5，因为一个点可能对应多张图片。5就表示这个点和五张图有关
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes( # (24, 115, 8, 184, 16, 169, 20, 226, 17, 154)这个里面混杂需要下面的处理
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2]))) # [24  8 16 20 17]这个点分别属于这些索引对应的图片中
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2]))) # [115 184 169 226 154] #这个点属于对应图片中的这个索引位置
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs) # 把上面得到的所有参数存在这个变量的对应索引下
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def main():
    if len(sys.argv) != 3:
        print("Usage: python read_model.py path/to/model/folder [.txt,.bin]")
        return

    cameras, images, points3D = read_model(path=sys.argv[1], ext=sys.argv[2])

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))


if __name__ == "__main__":
    main()

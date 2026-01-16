# 创建二进制形式的mnist数据集

为了收纳整齐，故将几个代码文件都放到了本目录下。每个代码文件都可以单独使用。要想使用，需要将代码文件挪到项目根目录中，它们会创建并使用`./dataset_downloaded/`中的内容。

## binary_mnist_create.py

下载mnist数据集到`./dataset_downloaded/mnist`，然后截取数据集的后五千张图片作为本次制作数据集的原材料。

因为原项目将mnist数据集的后五千张图片截取为验证集，其余用作训练集和测试机，而我们这次制作二进制形式的mnist是为了在新的环境中跑推理，为了避免在训练集上验证，还是遵循了原作者的截取做法。

为了确保“任意程序甚至C++也能轻易读取”，最通用的格式是Raw Binary（原始二进制）。这本质上就是将内存中的浮点数组直接Dump到硬盘上，没有头部信息（Header），没有压缩，C++里只需要一个fread就能读进去。

格式转换：本程序将每张图片的每个像素值都转为 float32，归一化到 [0.0, 1.0]，并将标签转为 One-hot。具体原理是，原mnist数据集的每个像素都是0-255的亮暗值，全部除以255后就可以归一化，归一化将方便训练和推理。

二进制存储：使用 numpy.tofile() 生成不带任何元数据的纯二进制文件。最后存储到了`dataset_downloaded/mnist_float`。标签以one_hot码的形式保存，5000张图片的标签都保存在同一个文件中。

## binary_mnist_check.py

尝试加载`binary_mnist_create.py`转换出的二进制mnist数据。检查是否出错。

## binary_mnist_check2png.py

尝试加载`binary_mnist_create.py`转换出的二进制mnist数据。更全面的检查，将一幅图从归一化恢复到亮暗像素，并生成png以供检查是否出错。

## binary_mnist_checkorigin.py

从原始mnist数据集中直接找出`binary_mnist_create.py`恢复制作的那张图片，以供比对。

## verify_specific_image.py

从原始mnist数据集中直接恢复出指定的一张图片，以供比对。
# Autopilot_Self_Driving_Car_Steering
自动驾驶之方向盘转动角度预测，基于keras，支持GPU加速。

The prediction of steering wheel angle of automatic driving, based on keras, runs on GPU.

本项目尝试使用神经网络的方式预测自动驾驶汽车的方向盘转动角度。

一般认为，车辆行驶的方向应为道路方向的切线方向，而车辆实际方向往往与切线方向有一定夹角。为了使车辆尽量沿着道路行驶，此时方向盘转动角度应使夹角角度减小。

本项目基于一个驾驶模拟器，手动驾驶可保存驾驶时的车前录像和同时刻的方向盘角度，生成数据集；自动驾驶模式可与运行中的 Python 脚本交互，实现无人驾驶。

## 数据<br>
神经网络预测方向盘转动角度，输入是车前录像，输出是方向盘转动角度，本质上是一个回归问题。道路车辆信息可以由车载摄像头采集，方向盘转动角度可以由车载电子系统采集。借助一个[模拟器](https://pan.baidu.com/s/1--8NRXVMdeV-jMUoimg4Zg)（提取码：8jyr）可以先研究清楚合适的神经网络架构。

这是模拟器截图：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E6%A8%A1%E6%8B%9F%E5%99%A8.jpg" alt="Sample"  width="400">
</p>

手动模式可以获取训练数据集：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E9%A9%BE%E9%A9%B6%E6%A8%A1%E5%BC%8F.jpg" alt="Sample"  width="400">
</p>

自动模式由程序控制，实现自动驾驶：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%87%AA%E5%8A%A8%E6%A8%A1%E5%BC%8F.jpg" alt="Sample"  width="400">
</p>


采集到的数据分为csv文件和IMG文件，csv文件是一个驾驶记录，包含了油门、方向、制动等信息，还存放了IMG文件的地址信息。IMG文件由虚拟的车载摄像头采集，分为左中右三个角度。用户可以手动模式采集数据（前提是要车技了得）。


项目文件分为train.py和drive.py。前者用于训练神经网络，后者是模拟器的输入接口，用于测试训练好的模型效果。


## 图像预处理、数据增强<br>
预处理包括以下几个方面：<br>
* 1.水平翻转<br>
* 2.亮度调整<br>
* 3.角度调整<br>
* 4.数据剔除<br>

为什么要水平翻转？如果右转弯样本远远多于左转弯样本，则数据不平衡，翻转操作减小了数据不平衡现象；<br>

为什么要调整亮度？实践发现亮度变化会严重影响神经网络判断方向，调整图像亮度可以提高网络应对不同环境的鲁棒性；<br>

角度调整是什么？当选取左侧或右侧摄像头画面时，需要将角度信息校准到车辆正前方，因此要做出角度调整；<br>

剔除什么数据？经分析，数据集中大部分数据为直行（转动角为0），严重影响数据平衡，要删除一部分。


## 模型结构<br>
本项目采用卷积神经网络结构，由卷积层和全连接层构成：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E9%A9%BE%E9%A9%B6.png" alt="Sample"  width="400">
</p>

读者可以根据硬件情况，适量增加层数、通道数、神经元数量。需要注意的是，要根据误差下降曲线，适量增加防止过拟合措施。


## 测试模型<br>
测试模型要用到drive.py，这是一个python和模拟器的接口脚本。具体操作步骤如下：

* 1.安装socketio、eventlet等依赖库。

* 2.将模拟器和drive.py、model.json、model.h5放在同一个文件夹下。

* 3.检查计算机有足够的显存，运行模拟器，进入Autonomous模式。

* 4.cmd中键入“ python drive.py model.json ”，随后模拟器中小车开始飞奔。

* 5.选中模拟器窗口按Esc退出运行，关闭cmd窗口。


## 测试结果<br>
这是我训练20个epoch之后得到的误差下降曲线：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/train_val_loss.jpg" alt="Sample"  width="400">
</p>


这是用模型驱动小车录制的结果：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/%E8%87%AA%E5%8A%A8%E9%A9%BE%E9%A9%B6.gif" alt="Sample"  width="400">
</p>

车辆可以顺利从起点行驶到终点，但是行驶过程并不平稳，乘坐体验可能不佳。经过分析，网络存在一定的过拟合问题，需要削减模型参数数量。

进一步的研究发现使用可分离卷积能够明显改善驾驶结果，同时减少一些计算量，[Using-EffNet-to-slim-models](https://github.com/LeeWise9/Using-EffNet-to-slim-models)。

改进后的模型的驾驶效果更加平滑，弯道的反应速度更灵敏：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/raw/master/drive_eff.gif" alt="Sample"  width="400">
</p>





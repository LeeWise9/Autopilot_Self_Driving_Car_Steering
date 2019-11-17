# Autopilot_Steering_Wheel_Angle_Prediction
自动驾驶之方向盘转动角度预测，基于keras，支持GPU加速。

The prediction of steering wheel angle of automatic driving, based on keras, runs on GPU.

自动驾驶中一个重要课题就是教会驾驶系统怎么甩盘子（方向盘），这就是驾驶方向预测。工业界主要有两种思路：传统计算或者神经网络，本项目介绍神经网络算法预测方向盘转动角度。


## 数据<br>
神经网络预测方向盘转动角度，输入端是道路车辆信息，输出端是方向盘转动角度。道路车辆信息可以由车载摄像头采集，方向盘转动角度可以由车载电子系统采集。当然还有更廉价的方式：借助一个[模拟器](https://pan.baidu.com/s/1--8NRXVMdeV-jMUoimg4Zg)（提取码：8jyr）。

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


采集到的数据分为csv文件和IMG文件，csv文件是一个驾驶记录，包含了油门、方向、制动等信息，还存放了IMG文件的地址信息。IMG文件由虚拟的车载摄像头采集，分为左中右三个角度。用户可以手动模式采集数据（前提是要车技了得），也可以用[现成的](https://pan.baidu.com/s/1237GDUMxKOhhUxwqsixt7Q)（提取码：tlpa）。


项目文件分为train.py和drive.py。前者用于训练神经网络，后者是模拟器的输入接口，用于测试训练好的模型效果。


## 图像预处理、数据增强<br>
预处理包括以下几个方面：<br>
1.水平翻转<br>
2.亮度调整<br>
3.角度调整<br>
4.数据剔除<br>

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

### Step1<br>
安装socketio、eventlet等依赖库。

### Step2<br>
将模拟器和drive.py、model.json、model.h5放在同一个文件夹下。

### Step3<br>
检查计算机有足够的显存，运行模拟器，进入Autonomous模式。

### Step4<br>
cmd中键入“ python drive.py model.json ”，随后模拟器中小车开始飞奔。

### Step5<br>
选中模拟器窗口按Esc退出运行，关闭cmd窗口。


## 测试结果<br>
这是我训练20个epoch之后得到的误差下降曲线：<br>
<p align="center">
	<img src="https://github.com/LeeWise9/Img_repositories/blob/master/train_val_loss.jpg" alt="Sample"  width="400">
</p>


这是用模型驱动小车录制的结果：<br>


这样开车可能要吃不少罚单吧 ╮(￣▽￣)╭


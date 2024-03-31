# ################################################################
# ## 2-4 尝试成本函数（损失）、优化器和/或激活函数的变体
# #################################################################
# print("################################################################")
# print("2-4 尝试成本函数（损失）、优化器和/或激活函数的变体")
# print("################################################################")
## 修改内容在 ## 5 定义损失函数

## 1 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

## 2 定义设备类型名称
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

## 3 定义数据集相关
batch_size = 128
transform = transforms.Compose( # 创建了一个转换组合，将一系列的数据预处理操作组合在一起。这里使用了两个预处理操作
    [transforms.ToTensor(), # 将图像转换为 PyTorch 张量，并将像素值缩放到 [0, 1] 的范围
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 对图像进行标准化处理，减去均值（0.5）并除以标准差（0.5）
# 创建了 CIFAR-10 数据集的训练集对象。root 参数指定了数据集存储的根目录，train=True 表示加载训练集，
# download=True 表示如果数据集不存在则自动下载，transform=transform 表示应用之前定义的数据预处理操作。
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 创建了一个训练集数据加载器。trainloader 负责从训练集中加载数据，shuffle=True 表示每个 epoch 都会对数据进行洗牌，
# num_workers=2 表示使用两个子进程来加载数据以加快速度。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# 这里分别创建了 CIFAR-10 数据集的测试集对象和测试集数据加载器。
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# 数据类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## 4 定义训练模型及参数统计参数函数
# 4-1 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def count_maxpool_operations(self, input, output, pool_layer):
        # input output 的维度均为 N,C,H,W
        kernel_maxpooling = pool_layer.kernel_size
        stride = pool_layer.stride
        padding = pool_layer.padding
        output_height = output.shape[2]
        output_width = output.shape[3]
        out_channels =  output.shape[1]
        # num_max
        # num_max = 0
        num_max = (output_height // kernel_maxpooling) * (output_width // kernel_maxpooling)* (kernel_maxpooling**2 -1)
        return num_max

    def count_conv_operations(self, input, output, output_pooled, conv_layer, pool_layer):
        # batch_size = input.size(0)
        out_channels, in_channels = output.size(1), conv_layer.in_channels
        output_height, output_width = output.size(2), output.size(3)
        filter_size = conv_layer.kernel_size[0]
        stride = conv_layer.stride[0]
        padding = conv_layer.padding[0]
        # Compute number of operations for convolution
        num_mults = output_height * output_width * in_channels * filter_size ** 2 * out_channels
        num_adds = output_height * output_width * in_channels * filter_size ** 2 * out_channels
        # num_maxs = output_height * output_width * out_channels
        num_maxs = self.count_maxpool_operations(output, output_pooled, pool_layer)
        total_ops = num_mults + num_adds + num_maxs
        return num_mults, num_adds, num_maxs, total_ops

    def count_operations(self, x):
        conv1_out = self.conv1(x)
        conv1_out_pooled = self.pool(F.relu(conv1_out))  # Apply max pooling after the first convolution
        conv2_out = self.conv2(conv1_out_pooled)
        conv2_out_pooled = self.pool(F.relu(conv2_out))  # Apply max pooling after the first convolution
        # Count operations for convolutional layer 1
        conv1_ops = self.count_conv_operations(x, conv1_out, conv1_out_pooled, self.conv1, self.pool)
        # Count operations for convolutional layer 2
        conv2_ops = self.count_conv_operations(conv1_out_pooled, conv2_out, conv2_out_pooled, self.conv2, self.pool)
        return conv1_ops, conv2_ops        

    def count_fc_operations(self, input, fc_layer):
        # Get the number of input features for the fully connected layer
        in_features = fc_layer.in_features
        # Get the number of output features for the fully connected layer
        out_features = fc_layer.out_features    
        # Compute number of operations for fully connected layer
        # print(str(out_features) + " * " + str(in_features))
        num_mults = out_features * in_features
        num_adds = out_features * in_features
        num_maxs = 0    
        total_ops = num_mults + num_adds
        return num_mults, num_adds, num_maxs, total_ops

    def count_total_operations(self, x):
        conv1_ops, conv2_ops = self.count_operations(x)
        fc1_ops = self.count_fc_operations(x, self.fc1)
        fc2_ops = self.count_fc_operations(x, self.fc2)
        fc3_ops = self.count_fc_operations(x, self.fc3)
        total_ops = sum(op[3] for op in [conv1_ops, conv2_ops, fc1_ops, fc2_ops, fc3_ops])
        return total_ops
# 4-2 实例化网络
net = Net().to(device)

## 5 定义损失函数
# 这行代码定义了损失函数，log likelihood损失函数
criterion = nn.NLLLoss()
# 这行代码定义了优化器，adam是一种常用的优化算法，
optimizer = optim.Adam(net.parameters(), lr=0.001) 

## 6 定义其他训练所使用的超参数
epoch_nums = 1   #为便于测试可定义为1，实际训练时根据需要改变

## 7 定义统计时间的辅助参数

## 8 训练模型
# 8-1 记录总时间_起始时间
if torch.cuda.is_available(): torch.cuda.synchronize()
total_time_start = time.time()
# 8-2 记录总操作数、单个epoch操作数、单个batch操作数
total_ops = 0
epoch_train_ops = 0
epoch_evaluate_ops = 0
batch_train_ops = 0
batch_evaluate_ops = 0
# 8-3 遍历epoch执行训练
for epoch in range(epoch_nums):
    print('\nEpoch: %d' % (epoch + 1))
    # 8-3-1 定义统计参数
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    iteration_num = 0
    # 8-3-2 遍历训练数据集，每次取batch_num个数据
    for i, data in enumerate(trainloader, 0):
        # 8-3-3 准备数据
        length = len(trainloader)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)        
        optimizer.zero_grad()
        # 8-3-4 统计训练时的操作数，仅需统计一次
        # images.shape[0] 是batch_size,count_total_operations在实际计算时未考虑batch_size
        if(epoch == 0 & i == 0):
            batch_train_ops = net.count_total_operations(inputs)
            batch_train_ops = inputs.shape[0] * batch_train_ops
        # 8-3-5 在数据输入网络前记录时间
        if torch.cuda.is_available(): torch.cuda.synchronize()
        batch_train_start = time.time()
        # 8-3-6 网络计算
        outputs = net(inputs)
        # 8-3-7 在网络处理完成后记录时间
        if torch.cuda.is_available(): torch.cuda.synchronize()
        batch_train_time = time.time() - batch_train_start
        # 8-3-8 计算loss及反向传播
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 8-3-9 每训练1个batch打印一次loss和准确率等参数
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        # 每秒运算次数
        batch_train_ops_per_second = batch_train_ops / batch_train_time
        print('[Epoch:%d, Iterations:%d] Loss: %.03f | Acc: %.3f%% | Ops: %d| Time: %.6fs | Ops/Sec : %d' % (
            epoch + 1, 
            (i + 1 + epoch * length), 
            sum_loss / (i + 1), 
            100. * correct / total,
            batch_train_ops,
            batch_train_time,
            batch_train_ops_per_second))
        iteration_num += 1
    if(epoch == 0):
        epoch_train_ops = batch_train_ops * iteration_num

    # 8-3-10 每训练完一个epoch测试一下准确率
    # test的时间相对于train较短，因此采用每个epoch统计一次时间
    print("Waiting Test!")
    with torch.no_grad():
        correct = 0
        total = 0
        iteration_num = 0
        # 在数据输入网络前记录时间
        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_evaluate_start = time.time()
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 统计训练时的操作数，仅需统计一次
            # images.shape[0] 是batch_size,count_total_operations在实际计算时未考虑batch_size
            if(epoch == 0 & i == 0):
                batch_evaluate_ops = net.count_total_operations(images)
                batch_evaluate_ops = images.shape[0] * batch_evaluate_ops
            outputs = net(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            iteration_num += 1
        acc = 100. * correct / total
        if(epoch == 0):
            epoch_evaluate_ops = batch_evaluate_ops * iteration_num 
        # 在网络处理完成后记录时间
        if torch.cuda.is_available(): torch.cuda.synchronize()
        epoch_evaluate_time = time.time() - epoch_evaluate_start
        # 每秒运算次数
        epoch_evaluate_ops_per_second = epoch_evaluate_time / epoch_evaluate_time
        print('[Epoch:%d] Validation Acc: %.3f%% | Ops: %d| Time: %.6fs | Ops/Sec : %d' % (
            epoch + 1, 
            acc,
            epoch_evaluate_ops,
            epoch_evaluate_time,
            epoch_evaluate_ops_per_second))
           

## 8-4 记录总时间_结束时间
if torch.cuda.is_available(): torch.cuda.synchronize()
total_time_end = time.time() - total_time_start
## 8-5 打印统计结果
total_ops = epoch_nums * (epoch_evaluate_ops + epoch_train_ops)
ops_per_second = total_ops / total_time_end
print('Time elapsed: %.3fs' % total_time_end)
print('total ops: %d' % total_ops)
print('ops/second : %d' % ops_per_second)
print('Finished Training')
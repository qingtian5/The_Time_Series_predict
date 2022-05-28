## 如何用自己的数据集训练时间序列模型

### 1. 克隆代码库，打开Time_sequence.ipynb，运行第一块代码，导入模块，如果没有用过d2l模块，可以先安装
```python
!pip install d2l
```

### 2. 再把数据转化为特定的格式
```python
datas : [11.29 11.47 11.16 ... 22.48 22.53 22.26]
```
代码中也有一个例子 ：
```python
with open('TRD_Dalyr.txt','r') as f:
    datas = f.readlines()
datas = [data.strip().split('\t') for data in datas]
datas = np.array(datas)
datas = datas[1:,2].astype(np.float)
print(datas)
```
如果要修改的话，可以直接修改这部分的代码，使得datas变成你自己的数据集。

### 3. 确定自己的模型参数
```python
tau = 30  # 用前30天的值预测今天的值
batch_size, n_train = 16, T-tau   # 只有前n_train个样本用于训练
epochs = 5   # 学习的轮数
lr = 0.01   # 学习率
```
### 4. 确定自己的模型架构
原来的模型架构为
```python
def get_net():
    net = nn.Sequential(nn.Linear(30, 50),
                        nn.ReLU(),
                        nn.Linear(50, 1))
    net.apply(init_weights)
    return net
```
以加一层线性层为例，新的模型架构修改为
```python
def get_net():
    net = nn.Sequential(nn.Linear(30, 50),
                        nn.ReLU(),
                        nn.Linear(50, 50),
                        nn.ReLU(),
                        nn.Linear(50, 1))
    net.apply(init_weights)
    return net
```

### 4. 画图的具体细节
第一个图是数据的具体分布，第二个图是单步预测，第三、四、五都是多步预测的结果图。
```python
# plt.savefig('data.svg')
```
取消注释$plt.savefig()$代码，就可以把画出来的图保存下来。

```python
max_steps = 64
#...
steps = (1, 4, 16, 64)
```
可以修改**max_steps**和**steps**的值来改变具体的步长和图像信息。

---
```python
help(d2l.plot) #查看d2l.plot帮助文档
Help on function plot in module d2l.torch:

plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None)
    Plot data points.
    
    Defined in :numref:`sec_calculus`
```
X,Y表示X轴和Y轴的数据,xlabel表示x轴的标签,ylabel表示y轴的标签,legend表示图例,xlim表示x轴的限制范围,ylim表示y轴的限制范围,figsize表示画布大小。

### 5. 最后运行代码即可

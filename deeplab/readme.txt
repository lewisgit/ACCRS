文件夹：
bin    : caffe, densecrf生成的可执行程序
config : 网络的配置文件
model  : 预训练好的文件
tmp    : 执行过程中生成的中间文件，最后会删除

文件：
test.py      ：手动调用，参数(源图像路径，生成的label存储路径)
train.py     ：手动调用，参数(源数据路径，label路径，指定具体训练数据的txt文件路径)
img_oper.py  : 为上面几个文件所调用，用于图像数据操作的

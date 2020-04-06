import numpy as np
import tensorflow as tf
from PIL import Image


def getTestPicArray(filename):
    im = Image.open(filename)
    x_s = 28
    y_s = 28
    out = im.resize((x_s, y_s), Image.ANTIALIAS)

    im_arr = np.array(out.convert('L'))

    num0 = 0
    num255 = 0
    threshold = 100

    for x in range(x_s):
        for y in range(y_s):
            if im_arr[x][y] > threshold:
                num255 = num255 + 1
            else:
                num0 = num0 + 1

    if (num255 > num0):
        print("convert!")
        for x in range(x_s):
            for y in range(y_s):
                im_arr[x][y] = 255 - im_arr[x][y]
                if (im_arr[x][y] < threshold):  im_arr[x][y] = 0
            # if(im_arr[x][y] > threshold) : im_arr[x][y] = 0
            # else : im_arr[x][y] = 255
            # if(im_arr[x][y] < threshold): im_arr[x][y] = im_arr[x][y] - im_arr[x][y] / 2

    out = Image.fromarray(np.uint8(im_arr))
    # out.save(filename.split('/')[0] + '/28pix/' + filename.split('/')[1])
    # print im_arr
    nm = im_arr.reshape((1, 784))

    nm = nm.astype(np.float32)
    nm = np.multiply(nm, 1.0 / 255.0)

    return nm


# im = Image.open('./images/test.png')
# data = list(im.getdata())
# result = [(255 - x) * 1.0 / 255.0 for x in data]
# 读取图片转成灰度格式
# img = Image.open('./images/test.png').convert('L')
#
# # resize的过程
# img = img.resize((28, 28))
#
# # 暂存像素值的一维数组
# result = []

# for i in range(28):
#     for j in range(28):
#         # mnist 里的颜色是0代表白色（背景），1.0代表黑色
#         pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
#         # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
#         result.append(pixel)
#
# result = np.array(result).reshape(28, 28)

# print(result)

data = getTestPicArray('./images/5.png')
result = np.reshape(data,(1,784))

# 为输入图像和目标输出类别创建节点
x = tf.placeholder("float", shape=[None, 784])  # 训练所需数据  占位符


# *************** 构建多层卷积网络 *************** #
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 取随机值，符合均值为0，标准差stddev为0.1
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1表示任意数量的样本数,大小为28x28，深度为1的张量

W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积在每个5x5的patch中算出32个特征。
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 在输出层之前加入dropout以减少过拟合
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 输出层
# tf.nn.softmax()将神经网络的输层变成一个概率分布
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

saver = tf.train.Saver()  # 定义saver

# *************** 开始识别 *************** #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "save/model.ckpt")  # 这里使用了之前保存的模型参数

    prediction = tf.argmax(y_conv, 1)
    predint = prediction.eval(feed_dict={x: result, keep_prob: 1.0}, session=sess)

    print("recognize result: %d" % predint[0])

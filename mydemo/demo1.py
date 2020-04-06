import matplotlib.pyplot as plt
import numpy as np

# 基本
# x = np.linspace(-1, 1, 50)
# y = 2 * x + 1
# y = x**2
# plt.plot(x, y)
# plt.show()

# figure
# x = np.linspace(-3, 3, 50)
# y1 = 2 * x + 1
# y2 = x**2
# plt.figure()
# plt.plot(x, y1)
# plt.figure(num=3, figsize=(8, 5))
# plt.plot(x, y1)
# plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--')
# plt.show()

# 坐标轴
x = np.linspace(-3, 3, 50)
y1 = 2 * x + 1
y2 = x ** 2

plt.figure(num=1, figsize=(8, 5))

plt.xlim((-1, 2))
plt.ylim((-1, 3))
plt.xlabel('x')
plt.ylabel('y')

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-1, 0, 1, 2, 3], ['reallybad', 'bad', 'normal', 'good', 'reallygood'])
plt.plot(x, y1, label='up')
plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--')
plt.legend(loc='upper right')

# gca = get current axis
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))


plt.show()

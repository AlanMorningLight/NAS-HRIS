import matplotlib.pyplot as plt
import numpy as np

def main():
    i=0
    y=[]
    x=[]
    with open(r'L:\NAS\NAS-RSI1\utils\result.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            x.append(i)
            y.append(float(line.split(' ')[7]))
            i = i+1
    plt.figure()
    # 定义x、y散点坐标
    x1 = np.array(x)
    y1 = np.array(y)
    print(x1)
    print(y1)
    # 用3次多项式拟合
    f1 = np.polyfit(x1, y1, 3)
    p1 = np.poly1d(f1)
    print(p1)  # 打印出拟合函数
    yvals1 = p1(x1)  # 拟合y值

    # 绘图
    plot1 = plt.plot(x1, y1, 's', label='original values')
    plot2 = plt.plot(x1, yvals1, 'r', label='polyfit values')

    plt.xlabel('epoch')
    plt.ylabel('MioU')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.title('GID_MioU')
    plt.savefig('Miou.png')
    plt.show()

if __name__ == '__main__':
    main()
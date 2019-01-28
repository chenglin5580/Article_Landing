# Illustration
"""
Model Supervised Learning
Lynn
2018.09.30

"""

# Package import
from Method import Para, MLP, Para2, MLP2
import time
from Eros import Eros as Asteriod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# para setting
method = '0'


# tensorboard --logdir="1 Eros/6 Time_Optimal_TrainNoModel/tensorboard"

def Controller_creat(train_flag_c=1, train_flag_g=1):
    para = Para(method,
                layer_num=6,
                units_num=512,
                activation_hidden=1,  # 1 tanh, 2 relu
                LR=0.0001,  # 学习率
                batch_size=1000,  # batch数量
                train=train_flag_c,  # 是否训练
                tensorboard=True,  # 是否存储tensorboard
                )
    controller = MLP(para=para)

    para2 = Para2(method,
                layer_num=6,
                units_num=512,
                activation_hidden=1,  # 1 tanh, 2 relu
                LR=0.0001,  # 学习率
                batch_size=1000,  # batch数量
                train=train_flag_g,  # 是否训练
                tensorboard=True,  # 是否存储tensorboard
                )
    Model = MLP2(para=para2)
    return controller, Model


def controller_verify():
    # 0：train
    # 1：verify
    # 2: 生成轨迹
    controller, model = Controller_creat(train_flag_c=1, train_flag_g=1)
    # model.verify_train()
    # model.verify_test()
    controller.verify_train()
    controller.verify_test()
    controller.verify_display()



def Get_Tra_Deviation():
    controller, model = Controller_creat(train_flag_c=1, train_flag_g=1)
    asteroid = Asteriod(model, controller, random=False)
    # asteroid.gravity_using = 'Nomodel'
    asteroid.gravity_using = 'NetworkModel'
    asteroid.epsilon_g = 1

    sample_tra = np.load('../1.1 Control_Samples/2 Learning/4 BenchControl/' + 'sample_tra.npy')
    sample_control = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_control.npy')
    sample_lambda = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_lambda.npy')
    sample_time_consume = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_time_consume.npy')


    iiii = 5
    i_s = 199 * iiii
    bench_tra = sample_tra[i_s:i_s + 199:1, :]
    bench_control = sample_control[i_s:i_s + 199:1, :]
    bench_lambda = sample_lambda[i_s:i_s + 199:1, :]
    bench_time = sample_time_consume[i_s:i_s + 199:1, :]

    state = bench_tra[0, :]
    tf = bench_time[0]
    asteroid.reset(state)
    asteroid.tf = tf

    end_flag = 0  # 时间终止
    # end_flag = 1 #距离终止

    Indirect_tra = asteroid.get_samples(op_variables=np.hstack((bench_lambda[0, :], bench_time[0]/asteroid.Tmax)))

    asteroid.reset(state)
    asteroid.tf = tf
    net_tra = asteroid.get_driven(end_flag=end_flag)

    np.savetxt('./result/' + 'bench_tra' + str(iiii) + '.txt', bench_tra)
    np.savetxt('./result/' + 'indirect_tra' + str(iiii) + '.txt', Indirect_tra["sample_tra"])
    np.savetxt('./result/' + 'net_tra' + str(iiii) + '.txt', net_tra["sample_tra"])

def Get_Statistic_Deviation():
    controller, model = Controller_creat(train_flag_c=1, train_flag_g=1)
    asteroid = Asteriod(model, controller, random=False)
    # asteroid.gravity_using = 'Nomodel'
    asteroid.gravity_using = 'NetworkModel'
    asteroid.epsilon_g = 1

    sample_tra = np.load('../1.1 Control_Samples/2 Learning/4 BenchControl/' + 'sample_tra.npy')
    sample_control = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_control.npy')
    sample_lambda = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_lambda.npy')
    sample_time_consume = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_time_consume.npy')


    deviations = [0, 0.5, 0.9, 1, 1.1, 1.5, 2]

    for deviation in deviations:

        bench_tra_terminal = np.zeros([0, 7])
        Indirect_tra_terminal = np.zeros([0, 7])
        net_tra_terminal = np.zeros([0, 7])

        asteroid.deviation = np.array(deviation)

        for iiii in range(30):
            print(deviation, iiii)
            i_s = 199 * iiii
            bench_tra = sample_tra[i_s:i_s + 199:1, :]
            bench_control = sample_control[i_s:i_s + 199:1, :]
            bench_lambda = sample_lambda[i_s:i_s + 199:1, :]
            bench_time = sample_time_consume[i_s:i_s + 199:1, :]

            state = bench_tra[0, :]
            tf = bench_time[0]
            asteroid.reset(state)
            asteroid.tf = tf

            end_flag = 0  # 时间终止
            # end_flag = 1 #距离终止

            Indirect_tra = asteroid.get_samples(op_variables=np.hstack((bench_lambda[0, :], bench_time[0]/asteroid.Tmax)))

            asteroid.reset(state)
            asteroid.tf = tf
            net_tra = asteroid.get_driven(end_flag=end_flag)

            bench_tra_terminal = np.vstack((bench_tra_terminal, bench_tra[-1, :]))
            Indirect_tra_terminal = np.vstack((Indirect_tra_terminal, Indirect_tra["sample_tra"][-1, :]))
            terminal_position = Indirect_tra["sample_tra"][-1, 0:3]
            terminal_position[1] += 3.7
            print("Indirect", np.linalg.norm(terminal_position))

            net_tra_terminal = np.vstack((net_tra_terminal, net_tra["sample_tra"][-1, :]))
            terminal_position = net_tra["sample_tra"][-1, 0:3]
            terminal_position[1] += 3.7
            print("Net", np.linalg.norm(terminal_position))

        np.savetxt('./result/' + 'bench_tra_terminal'+ str(deviation) + '.txt', bench_tra_terminal)
        np.savetxt('./result/' + 'Indirect_tra_terminal'+ str(deviation) + '.txt', Indirect_tra_terminal)
        np.savetxt('./result/' + 'net_tra_terminal'+ str(deviation) + '.txt', net_tra_terminal)


def robustness_display():

    deviations = [0, 0.5, 0.9, 1, 1.1, 1.5, 2]
    Indirect_position = np.empty([30, 0])
    Indirect_velocity = np.empty([30, 0])
    Net_position = np.empty([30, 0])
    Net_velocity = np.empty([30, 0])

    for deviation in deviations:
        bench_tra_terminal, Indirect_tra_terminal, net_tra_terminal = [],[],[]
        bench_tra_terminal = np.loadtxt('./result/' + 'bench_tra_terminal' + str(deviation) + '.txt')
        bench_tra_terminal[:, 1] += 3.7
        Indirect_tra_terminal = np.loadtxt('./result/' + 'Indirect_tra_terminal' + str(deviation) + '.txt')
        Indirect_tra_terminal[:, 1] += 3.7
        net_tra_terminal = np.loadtxt('./result/' + 'net_tra_terminal' + str(deviation) + '.txt')
        net_tra_terminal[:, 1] += 3.7


        position_error = np.linalg.norm(Indirect_tra_terminal[-30:, 0:3], axis=1)
        Indirect_position = np.hstack((Indirect_position, position_error[:, np.newaxis]))

        velocity_error = np.linalg.norm(Indirect_tra_terminal[-30:, 3:6], axis=1)
        Indirect_velocity = np.hstack((Indirect_velocity, velocity_error[:, np.newaxis]))

        position_error = np.linalg.norm(net_tra_terminal[-30:, 0:3], axis=1)
        Net_position = np.hstack((Net_position, position_error[:, np.newaxis]))


        velocity_error = np.linalg.norm(net_tra_terminal[-30:, 3:6], axis=1)
        Net_velocity = np.hstack((Net_velocity, velocity_error[:, np.newaxis]))


    labels = [r'$-100\%$', r'$-50\%$', r'$10\%$', r'$0$', r'$+10\%$', r'$+50\%$', r'$+100\%$']


    plt.figure(1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 3.84))
    # rectangular box plot
    bplot1 = axes[0].boxplot(Indirect_position,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             showfliers=False,  # 异常值
                             labels=labels)  # will be used to label x-ticks
    axes[0].set_title('Position Error of Indirect Method ', fontsize=12)

    # notch shape box plot
    bplot2 = axes[1].boxplot(Indirect_velocity,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             showfliers=False,
                             labels=labels)  # will be used to label x-ticks
    axes[1].set_title('Velocity Error of Indirect Method ', fontsize=12)
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'blue', 'aliceblue', 'lightblue', 'pink']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    for ax in axes:
        ax.yaxis.grid(True)
        # ax.set_xlabel('Three separate samples')
    axes[0].set_ylabel(r'Position Error (km)', fontsize=12)
    axes[1].set_ylabel(r'Velocity Error (km/s)', fontsize=12)

    # plt.savefig('box_plot_control.png', dpi=720)

    plt.figure(2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 3.84))
    # rectangular box plot
    bplot1 = axes[0].boxplot(Net_position/8,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             showfliers=False,  # 异常值
                             labels=labels)  # will be used to label x-ticks
    axes[0].set_title('Position Error of DNN-based control ', fontsize=12)

    # notch shape box plot
    bplot2 = axes[1].boxplot(Net_velocity/3,
                             notch=True,  # notch shape
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             showfliers=False,
                             labels=labels)  # will be used to label x-ticks
    axes[1].set_title('Velocity Error of DNN-based control', fontsize=12)
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'blue', 'aliceblue', 'lightblue', 'pink']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    for ax in axes:
        ax.yaxis.grid(True)
        # ax.set_xlabel('Three separate samples')
    axes[0].set_ylabel(r'Position Error (km)', fontsize=12)
    axes[1].set_ylabel(r'Velocity Error (km/s)', fontsize=12)


    plt.show()



def robustness_statistis():

    deviations = [0, 0.5, 0.9, 1, 1.1, 1.5, 2]

    for deviation in deviations:
        bench_tra_terminal, Indirect_tra_terminal, net_tra_terminal = [],[],[]
        bench_tra_terminal = np.loadtxt('./result/' + 'bench_tra_terminal' + str(deviation) + '.txt')
        bench_tra_terminal[:, 1] += 3.7
        Indirect_tra_terminal = np.loadtxt('./result/' + 'Indirect_tra_terminal' + str(deviation) + '.txt')
        Indirect_tra_terminal[:, 1] += 3.7
        net_tra_terminal = np.loadtxt('./result/' + 'net_tra_terminal' + str(deviation) + '.txt')
        net_tra_terminal[:, 1] += 3.7




        print("------------deviation-------------", deviation)

        position_error = np.linalg.norm(Indirect_tra_terminal[:, 0:3], axis=1)
        velocity_error = np.linalg.norm(Indirect_tra_terminal[:, 3:6], axis=1)

        print("Indirect position_error mean", np.mean(position_error))
        print("Indirect position_error max", np.max(position_error))

        print("Indirect velocity_error mean", np.mean(velocity_error))
        print("Indirect velocity_error max", np.max(velocity_error))

        position_error = np.linalg.norm(net_tra_terminal[:, 0:3], axis=1)/8
        velocity_error = np.linalg.norm(net_tra_terminal[:, 3:6], axis=1)/3

        print("Net position_error mean", np.mean(position_error))
        print("Net position_error max", np.max(position_error))

        print("Net velocity_error mean", np.mean(velocity_error))
        print("Net velocity_error max", np.max(velocity_error))


def test():
    controller, model = Controller_creat(train_flag_c=1, train_flag_g=1)
    asteroid = Asteriod(model, controller, random=False)
    # asteroid.gravity_using = 'Nomodel'
    asteroid.gravity_using = 'NetworkModel'
    asteroid.epsilon_g = 1

    sample_tra = np.load('../1.1 Control_Samples/2 Learning/4 BenchControl/' + 'sample_tra.npy')
    sample_control = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_control.npy')
    sample_lambda = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_lambda.npy')
    sample_time_consume = np.load('../1.1 Control_Samples/1 Generation/Samples/' + 'sample_time_consume.npy')

    deviations = [1]

    for deviation in deviations:

        bench_tra_terminal = np.zeros([0, 7])
        Indirect_tra_terminal = np.zeros([0, 7])
        net_tra_terminal = np.zeros([0, 7])

        asteroid.deviation = np.array(deviation)

        for iiii in range(4):
            print(deviation, iiii)
            i_s = 199 * iiii
            bench_tra = sample_tra[i_s:i_s + 199:1, :]
            bench_control = sample_control[i_s:i_s + 199:1, :]
            bench_lambda = sample_lambda[i_s:i_s + 199:1, :]
            bench_time = sample_time_consume[i_s:i_s + 199:1, :]

            state = bench_tra[0, :]
            tf = bench_time[0]
            asteroid.reset(state)
            asteroid.tf = tf

            end_flag = 0  # 时间终止
            # end_flag = 1 #距离终止

            Indirect_tra = asteroid.get_samples(
                op_variables=np.hstack((bench_lambda[0, :], bench_time[0] / asteroid.Tmax)))

            asteroid.reset(state)
            asteroid.tf = tf
            net_tra = asteroid.get_driven(end_flag=end_flag)

            bench_tra_terminal = np.vstack((bench_tra_terminal, bench_tra[-1, :]))
            Indirect_tra_terminal = np.vstack((Indirect_tra_terminal, Indirect_tra["sample_tra"][-1, :]))
            terminal_position = Indirect_tra["sample_tra"][-1, 0:3]
            terminal_position[1] += 3.7
            print("Indirect", np.linalg.norm(terminal_position))

            net_tra_terminal = np.vstack((net_tra_terminal, net_tra["sample_tra"][-1, :]))
            terminal_position = net_tra["sample_tra"][-1, 0:3]
            terminal_position[1] += 3.7
            print("Net", np.linalg.norm(terminal_position))

            print(net_tra["sample_tra"][-1, :])



        np.savetxt('./result/' + 'testbench_tra_terminal' + str(deviation) + '.txt', bench_tra_terminal)
        np.savetxt('./result/' + 'testIndirect_tra_terminal' + str(deviation) + '.txt', Indirect_tra_terminal)
        np.savetxt('./result/' + 'testnet_tra_terminal' + str(deviation) + '.txt', net_tra_terminal)





if __name__ == '__main__':
    ## 函数
    # controller_train()    # 训练控制器在train_data上进行训练
    # controller_verify()     # 训练控制器在train_data和test_data上进行测试
    # Get_Tra_Deviation() # 参数拉偏下获得三种轨迹
    # Get_Statistic_Deviation()
    # robustness_display()
    # robustness_statistis()
    test()
#


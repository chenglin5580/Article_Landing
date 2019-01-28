
# Illustration
"""
Model Supervised Learning
Lynn
2018.09.30

"""

# Package Import
import tensorflow as tf
import numpy as np
import os
import sys
from os import path
from sklearn import preprocessing
import matplotlib.pyplot as plt

# tf.set_random_seed(2)

class Object(object):

    def __init__(self):
        pass

class Para(object):
    def __init__(
            self,
            method,
            layer_num=6,
            units_num=256,
            activation_hidden=1,
            LR=0.0001,       # 学习率
            batch_size=256,  # 批次数量
            train=True,  # 训练的时候有探索
            tensorboard=True,
    ):
        self.method = method

        # train info
        self.trainInfo = Object()
        self.trainInfo.LR = LR
        self.trainInfo.tensorboard = tensorboard
        if train == 0:
            self.trainInfo.flag = True
        else:
            self.trainInfo.flag = False
        self.trainInfo.iteration = 0
        self.trainInfo.batch_size = batch_size

        # Obect info
        self.objectInfo = Object()
        self.objectInfo.r_dim = 3
        self.objectInfo.G_dim = 3

        # model info
        self.modelInfo = Object()
        self.modelInfo.layer_num = layer_num
        self.modelInfo.units_num = units_num
        self.modelInfo.activation_hidden = activation_hidden

        # data info
        ## data load path
        self.dataInfo = Object()
        d = path.dirname(__file__)  # 返回当前文件所在的目录
        # parent_path = path.dirname(d)  # 获得d所在的目录,即d的父级目录
        # data_path = d + '/1.1 Control_Samples/2 Learning/2 Training_Data/'
        data_path = '../1.1 Control_Samples/2 Learning/2 Training_Data/'
        self.dataInfo.train_data_path = data_path
        # self.dataInfo.test_data_path = data_path + '/test_data.npy'
        ## model save
        model_path0 = os.path.join(sys.path[0], 'controller')
        if not os.path.exists(model_path0):
            os.mkdir(model_path0)
        self.dataInfo.model_save_path = os.path.join(model_path0, 'data.chkp')
        ## tensorboard save
        self.dataInfo.log_save_path = './tensorboard_con/' + str(layer_num)+'/' + str(units_num) +'/'  + str(LR) +'/' + str(batch_size) +'/' \
                                    + str(activation_hidden) +'/' + self.method


class DataLoader(object):
    def __init__(self, para):
        # train data 构建
        self.para =  para
        self.train_data_x = np.load(self.para.dataInfo.train_data_path+"train_data_x.npy")
        self.train_data_y = np.load(self.para.dataInfo.train_data_path + "train_data_y.npy")
        self.test_data_x = np.load(self.para.dataInfo.train_data_path + "test_data_x.npy")
        self.test_data_y = np.load(self.para.dataInfo.train_data_path + "test_data_y.npy")
        self.train_data_size = len(self.train_data_x[:, 0])
        self.test_data_size = len(self.test_data_x[:, 0])


        # train data 构建
        self.DataPreProcess()
        self.train_data_x = self.data_transform_x.transform(self.train_data_x)
        self.test_data_x = self.data_transform_x.transform(self.test_data_x)
        self.train_data_y = self.data_transform_y.transform(self.train_data_y)
        self.test_data_y = self.data_transform_y.transform(self.test_data_y)


    def get_batch(self):
        # get batch
        index = np.random.randint(0, np.shape(self.train_data_x)[0], self.para.trainInfo.batch_size)
        return self.train_data_x[index, :], self.train_data_y[index, :]

    def get_batch_test(self):
        # get batch
        index = np.random.randint(0, np.shape(self.test_data_x)[0], self.para.trainInfo.batch_size)
        return self.test_data_x[index, :], self.test_data_y[index, :]


    def DataPreProcess(self):
        # feature extraction
        self.data_transform_x = preprocessing.MinMaxScaler([-1, 1])
        self.data_transform_x.fit_transform(self.train_data_x)
        self.data_transform_y = preprocessing.MinMaxScaler([-1, 1])
        self.data_transform_y.fit_transform(self.train_data_y)



class MLP(object):
    def __init__(self, para):
        self.para = para
        self.data_loader = DataLoader(para=self.para)
        con_graph = tf.Graph()

        self.sess = tf.Session(graph=con_graph)

        with con_graph.as_default():
            # placeholder
            self.state7_input = tf.placeholder(tf.float32, [None, 7], 'state')
            self.alpha1_input = tf.placeholder(tf.float32, [None, 1], 'Gravity1')
            self.alpha2_input = tf.placeholder(tf.float32, [None, 1], 'Gravity2')
            self.alpha3_input = tf.placeholder(tf.float32, [None, 1], 'Gravity3')
            self.time_con_input = tf.placeholder(tf.float32, [None, 1], 'time_con')
            self.mass_con_input = tf.placeholder(tf.float32, [None, 1], 'mass_con')

            # build network
            if self.para.modelInfo.activation_hidden == 1:
                af_hidden = tf.nn.tanh
            elif self.para.modelInfo.activation_hidden == 2:
                af_hidden = tf.nn.relu
            else:
                af_hidden = None
                print("ActivationFuntion Setting Error")

            netname = "alpha1_net"
            with tf.variable_scope(netname):
                self.alpha1_pred = self._build_net(self.state7_input, scope='eval', trainable=True,
                                                   af_hidden=af_hidden, af_out=tf.nn.tanh, num_out=1)
                tf.summary.histogram(netname +'/eval', self.alpha1_pred)
                self.alpha1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.alpha1_error = tf.subtract(self.alpha1_input, self.alpha1_pred, name=netname + 'error')
                self.alpha1_MSE = tf.losses.mean_squared_error(labels=self.alpha1_input, predictions=self.alpha1_pred)
                self.alpha1_MAE = tf.reduce_mean(tf.abs(self.alpha1_error))
                self.alpha1_EMax = tf.norm(tf.abs(self.alpha1_error), ord=np.inf)
                tf.summary.scalar(netname +'MSE', self.alpha1_MSE)
                tf.summary.scalar(netname +'MAE', self.alpha1_MAE)
                tf.summary.scalar(netname +'Max', self.alpha1_EMax)
                self.alpha1_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.alpha1_MSE,
                                                                                            var_list=self.alpha1_params)

            netname = "alpha2_net"
            with tf.variable_scope(netname):
                self.alpha2_pred = self._build_net(self.state7_input, scope='eval', trainable=True,
                                                   af_hidden=af_hidden, af_out=tf.nn.tanh, num_out=1)
                tf.summary.histogram(netname + '/eval', self.alpha2_pred)
                self.alpha2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.alpha2_error = tf.subtract(self.alpha2_input, self.alpha2_pred, name=netname + 'error')
                self.alpha2_MSE = tf.losses.mean_squared_error(labels=self.alpha2_input, predictions=self.alpha2_pred)
                self.alpha2_MAE = tf.reduce_mean(tf.abs(self.alpha2_error))
                self.alpha2_EMax = tf.norm(tf.abs(self.alpha2_error), ord=np.inf)
                tf.summary.scalar(netname + 'MSE', self.alpha2_MSE)
                tf.summary.scalar(netname + 'MAE', self.alpha2_MAE)
                tf.summary.scalar(netname + 'Max', self.alpha2_EMax)
                self.alpha2_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.alpha2_MSE,
                                                                                            var_list=self.alpha2_params)

            netname = "alpha3_net"
            with tf.variable_scope(netname):
                self.alpha3_pred = self._build_net(self.state7_input, scope='eval', trainable=True,
                                                   af_hidden=af_hidden, af_out=tf.nn.tanh, num_out=1)
                tf.summary.histogram(netname + '/eval', self.alpha3_pred)
                self.alpha3_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.alpha3_error = tf.subtract(self.alpha3_input, self.alpha3_pred, name=netname + 'error')
                self.alpha3_MSE = tf.losses.mean_squared_error(labels=self.alpha3_input, predictions=self.alpha3_pred)
                self.alpha3_MAE = tf.reduce_mean(tf.abs(self.alpha3_error))
                self.alpha3_EMax = tf.norm(tf.abs(self.alpha3_error), ord=np.inf)
                tf.summary.scalar(netname + 'MSE', self.alpha3_MSE)
                tf.summary.scalar(netname + 'MAE', self.alpha3_MAE)
                tf.summary.scalar(netname + 'Max', self.alpha3_EMax)
                self.alpha3_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.alpha3_MSE,
                                                                                            var_list=self.alpha3_params)

            netname = "time_con_net"
            with tf.variable_scope(netname):
                self.time_con_pred = self._build_net(self.state7_input, scope='eval', trainable=True,
                                                   af_hidden=af_hidden, af_out=None, num_out=1)
                tf.summary.histogram(netname + '/eval', self.time_con_pred)
                self.time_con_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.time_con_error = tf.subtract(self.time_con_input, self.time_con_pred, name=netname + 'error')
                self.time_con_MSE = tf.losses.mean_squared_error(labels=self.time_con_input, predictions=self.time_con_pred)
                self.time_con_MAE = tf.reduce_mean(tf.abs(self.time_con_error))
                self.time_con_EMax = tf.norm(tf.abs(self.time_con_error), ord=np.inf)
                tf.summary.scalar(netname + 'MSE', self.time_con_MSE)
                tf.summary.scalar(netname + 'MAE', self.time_con_MAE)
                tf.summary.scalar(netname + 'Max', self.time_con_EMax)
                self.time_con_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.time_con_MSE,
                                                                                            var_list=self.time_con_params)

            netname = "mass_con_net"
            with tf.variable_scope(netname):
                self.mass_con_pred = self._build_net(self.state7_input, scope='eval', trainable=True,
                                                     af_hidden=af_hidden, af_out=None, num_out=1)
                tf.summary.histogram(netname + '/eval', self.mass_con_pred)
                self.mass_con_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.mass_con_error = tf.subtract(self.mass_con_input, self.mass_con_pred, name=netname + 'error')
                self.mass_con_MSE = tf.losses.mean_squared_error(labels=self.mass_con_input,
                                                                 predictions=self.mass_con_pred)
                self.mass_con_MAE = tf.reduce_mean(tf.abs(self.mass_con_error))
                self.mass_con_EMax = tf.norm(tf.abs(self.mass_con_error), ord=np.inf)
                tf.summary.scalar(netname + 'MSE', self.mass_con_MSE)
                tf.summary.scalar(netname + 'MAE', self.mass_con_MAE)
                tf.summary.scalar(netname + 'Max', self.mass_con_EMax)
                self.mass_con_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.mass_con_MSE,
                                                                                              var_list=self.mass_con_params)


            self.actor_saver = tf.train.Saver()
        # sess initialization

        if self.para.trainInfo.flag:
            with self.sess.as_default():
                with con_graph.as_default():
                    self.sess.run(tf.global_variables_initializer())
        else:
            with self.sess.as_default():
                with con_graph.as_default():
                    self.actor_saver.restore(self.sess, self.para.dataInfo.model_save_path)

        # tensorboard save
        if self.para.trainInfo.flag and self.para.trainInfo.tensorboard:
            with self.sess.as_default():
                with con_graph.as_default():
                    self.merged = tf.summary.merge_all()
                    self.writer = tf.summary.FileWriter(self.para.dataInfo.log_save_path, self.sess.graph)

    def model_save(self):
        self.actor_saver.save(self.sess, self.para.dataInfo.model_save_path)

    def _build_net(self, s, scope=None, trainable=True, af_hidden=tf.nn.relu, af_out=None, num_out=1):
        # 建立actor网络
        layer_num = self.para.modelInfo.layer_num
        units_num = self.para.modelInfo.units_num
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, units_num, activation=af_hidden, name='l0', trainable=trainable)
            for i in range(layer_num-2):
                net = tf.layers.dense(net, units_num, activation=af_hidden, name='l'+str(i+1), trainable=trainable)
            a = tf.layers.dense(net, num_out, activation=af_out, name='a', trainable=trainable)
            return a

    def control_predict(self, r):
        if r.ndim == 1:
            s = r[np.newaxis, :].copy()
        else:
            s = r.copy()
        len_output = len(s[:, 0])
        s = self.data_loader.data_transform_x.transform(s)
        alpha1, alpha2, alpha3 = self.sess.run([self.alpha1_pred, self.alpha2_pred, self.alpha3_pred], {self.state7_input: s})
        temp_y = self.data_loader.test_data_y[0:len_output:1, :]
        temp_y[:, 9:10] = alpha1
        temp_y[:, 10:11] = alpha2
        temp_y[:, 11:12] = alpha3
        temp_y = self.data_loader.data_transform_y.inverse_transform(temp_y)
        control = temp_y[:, 9:12:1]
        return control

    def time_con_predict(self, r):
        if r.ndim == 1:
            s = r[np.newaxis, :].copy()
        else:
            s = r.copy()
        len_output = len(s[:, 0])
        s = self.data_loader.data_transform_x.transform(s)
        time_con_pred = self.sess.run([self.time_con_pred], {self.state7_input: s})
        temp_y = self.data_loader.test_data_y[0:len_output:1, :]
        temp_y[:, 13:14] = time_con_pred[0]
        temp_y = self.data_loader.data_transform_y.inverse_transform(temp_y)
        time_con_pred = temp_y[:, 13:14:1]
        return time_con_pred

    def mass_con_predict(self, r):
        if r.ndim == 1:
            s = r[np.newaxis, :].copy()
        else:
            s = r.copy()
        len_output = len(s[:, 0])
        s = self.data_loader.data_transform_x.transform(s)
        mass_con_pred = self.sess.run([self.mass_con_pred], {self.state7_input: s})
        temp_y = self.data_loader.test_data_y[0:len_output:1, :]
        temp_y[:, 12:13] = mass_con_pred[0]
        temp_y = self.data_loader.data_transform_y.inverse_transform(temp_y)
        mass_con_pred = temp_y[:, 12:13:1]
        return mass_con_pred

    def train(self):
        batch_x, batch_y = self.data_loader.get_batch()
        state7_batch = batch_x
        alpha1_batch = batch_y[:, 9:10]
        alpha2_batch = batch_y[:, 10:11]
        alpha3_batch = batch_y[:, 11:12]
        time_con_batch = batch_y[:, 12:13]
        mass_con_batch = batch_y[:, 13:14]

        self.sess.run([self.alpha1_Train, self.alpha2_Train, self.alpha3_Train, self.time_con_Train,
                       self.mass_con_Train],
                      {self.state7_input: state7_batch,
                       self.alpha1_input: alpha1_batch,
                       self.alpha2_input: alpha2_batch,
                       self.alpha3_input: alpha3_batch,
                       self.time_con_input: time_con_batch,
                       self.mass_con_input: mass_con_batch,
                       })

        if self.para.trainInfo.tensorboard:
            if self.para.trainInfo.iteration % 100 == 0:
                result_merge = self.sess.run(self.merged,
                                           {self.state7_input: state7_batch,
                                            self.alpha1_input: alpha1_batch,
                                            self.alpha2_input: alpha2_batch,
                                            self.alpha3_input: alpha3_batch,
                                            self.time_con_input: time_con_batch,
                                            self.mass_con_input: mass_con_batch,
                                            })
                self.writer.add_summary(result_merge, self.para.trainInfo.iteration)
                print('Train', '-----------------', self.para.trainInfo.iteration)
                print('MAE', self.sess.run([self.alpha1_MAE, self.alpha2_MAE, self.alpha3_MAE, self.time_con_MAE,
                                            self.mass_con_MAE],
                                           {self.state7_input: state7_batch,
                                            self.alpha1_input: alpha1_batch,
                                            self.alpha2_input: alpha2_batch,
                                            self.alpha3_input: alpha3_batch,
                                            self.time_con_input: time_con_batch,
                                            self.mass_con_input: mass_con_batch,
                                            }))
        self.para.trainInfo.iteration += 1

    def verify_train(self):
        batch_x, batch_y = self.data_loader.get_batch()
        state7_batch = batch_x
        alpha1_batch = batch_y[:, 9:10]
        alpha2_batch = batch_y[:, 10:11]
        alpha3_batch = batch_y[:, 11:12]
        time_con_batch = batch_y[:, 12:13]
        mass_con_batch = batch_y[:, 13:14]

        print('Train MAE 1 2 3', self.sess.run([self.alpha1_MAE, self.alpha2_MAE, self.alpha3_MAE, self.time_con_MAE,
                                            self.mass_con_MAE],
                                           {self.state7_input: state7_batch,
                                            self.alpha1_input: alpha1_batch,
                                            self.alpha2_input: alpha2_batch,
                                            self.alpha3_input: alpha3_batch,
                                            self.time_con_input: time_con_batch,
                                            self.mass_con_input: mass_con_batch,
                                            }))




    def verify_test(self):

        batch_x, batch_y = self.data_loader.get_batch_test()
        state7_batch = batch_x
        alpha1_batch = batch_y[:, 9:10]
        alpha2_batch = batch_y[:, 10:11]
        alpha3_batch = batch_y[:, 11:12]
        time_con_batch = batch_y[:, 12:13]
        mass_con_batch = batch_y[:, 13:14]

        print('Test MAE 1 2 3', self.sess.run([self.alpha1_MAE, self.alpha2_MAE, self.alpha3_MAE, self.time_con_MAE,
                                            self.mass_con_MAE],
                                           {self.state7_input: state7_batch,
                                            self.alpha1_input: alpha1_batch,
                                            self.alpha2_input: alpha2_batch,
                                            self.alpha3_input: alpha3_batch,
                                            self.time_con_input: time_con_batch,
                                            self.mass_con_input: mass_con_batch,
                                            }))


    def verify_display(self):

        batch_x, batch_y = self.data_loader.get_batch()
        state7_batch = batch_x
        alpha1_batch = batch_y[:, 9:10]
        alpha2_batch = batch_y[:, 10:11]
        alpha3_batch = batch_y[:, 11:12]
        time_con_batch = batch_y[:, 12:13]
        mass_con_batch = batch_y[:, 13:14]

        print('Train MAE 1 2 3', self.sess.run([self.alpha1_MAE, self.alpha2_MAE, self.alpha3_MAE, self.time_con_MAE,
                                                self.mass_con_MAE],
                                               {self.state7_input: state7_batch,
                                                self.alpha1_input: alpha1_batch,
                                                self.alpha2_input: alpha2_batch,
                                                self.alpha3_input: alpha3_batch,
                                                self.time_con_input: time_con_batch,
                                                self.mass_con_input: mass_con_batch,
                                                }))
        alpha1_pred, alpha2_pred, alpha3_pred, time_con_pred, mass_con_pred = \
            self.sess.run(
                [self.alpha1_pred, self.alpha2_pred, self.alpha3_pred, self.time_con_pred, self.mass_con_pred],
                {self.state7_input: state7_batch, })

        print('Train Prediction error',
              np.hstack((alpha1_pred - alpha1_batch, alpha2_pred - alpha2_batch, alpha3_pred - alpha3_batch,
                         time_con_pred - time_con_batch, mass_con_pred - mass_con_batch)))

        batch_pred = np.copy(batch_y)
        batch_pred[:, 9:10] = alpha1_pred
        batch_pred[:, 10:11] = alpha2_pred
        batch_pred[:, 11:12] = alpha3_pred
        batch_pred[:, 12:13] = time_con_pred
        batch_pred[:, 13:14] = mass_con_pred

        batch_pred_origin = self.data_loader.data_transform_y.inverse_transform(batch_pred)
        batch_y_origin = self.data_loader.data_transform_y.inverse_transform(batch_y)

        print('Train Origin error', batch_pred_origin[:, 9:14] - batch_y_origin[:, 9:14])

        # norm222 = np.linalg.norm(batch_pred_origin[:, 9:14] - batch_y_origin[:, 9:14], axis=1)

        all_data1 = (batch_pred_origin[:, 9:14] - batch_y_origin[:, 9:14]) * 100

        print('sss', all_data1[:, -2].max())

        all_data1[:, -2] = all_data1[:, -2]/all_data1[:, -2].max()
        all_data1[:, -1] = all_data1[:, -1] / all_data1[:, -1].max()

        all_data1 -= np.mean(all_data1, axis=0)


        labels = [r'$\alpha_1$', r'$\alpha_2$', r'$\alpha_3$', r'$t_f$', r'$\Delta m$']

        batch_x, batch_y = self.data_loader.get_batch_test()
        state7_batch = batch_x
        alpha1_batch = batch_y[:, 9:10]
        alpha2_batch = batch_y[:, 10:11]
        alpha3_batch = batch_y[:, 11:12]
        time_con_batch = batch_y[:, 12:13]
        mass_con_batch = batch_y[:, 13:14]

        print('Train MAE 1 2 3', self.sess.run([self.alpha1_MAE, self.alpha2_MAE, self.alpha3_MAE, self.time_con_MAE,
                                            self.mass_con_MAE],
                                           {self.state7_input: state7_batch,
                                            self.alpha1_input: alpha1_batch,
                                            self.alpha2_input: alpha2_batch,
                                            self.alpha3_input: alpha3_batch,
                                            self.time_con_input: time_con_batch,
                                            self.mass_con_input: mass_con_batch,
                                            }))
        alpha1_pred, alpha2_pred, alpha3_pred, time_con_pred,  mass_con_pred = \
            self.sess.run([self.alpha1_pred,  self.alpha2_pred, self.alpha3_pred, self.time_con_pred, self.mass_con_pred],
                                                              {self.state7_input: state7_batch,})

        print('Train Prediction error',
              np.hstack((alpha1_pred - alpha1_batch, alpha2_pred - alpha2_batch, alpha3_pred - alpha3_batch,
                         time_con_pred-time_con_batch, mass_con_pred-mass_con_batch)))

        batch_pred = np.copy(batch_y)
        batch_pred[:, 9:10] = alpha1_pred
        batch_pred[:, 10:11] = alpha2_pred
        batch_pred[:, 11:12] = alpha3_pred
        batch_pred[:, 12:13] = time_con_pred
        batch_pred[:, 13:14] = mass_con_pred

        batch_pred_origin = self.data_loader.data_transform_y.inverse_transform(batch_pred)
        batch_y_origin = self.data_loader.data_transform_y.inverse_transform(batch_y)

        print('Train Origin error', batch_pred_origin[:, 9:14] - batch_y_origin[:, 9:14])

        all_data2 = (batch_pred_origin[:, 9:14] - batch_y_origin[:, 9:14]) * 100

        print('sss', all_data2[:, -2].max())

        all_data2[:, -2] = all_data2[:, -2] / all_data2[:, -2].max()
        all_data2[:, -1] = all_data2[:, -1] / all_data2[:, -1].max()

        all_data2 -= np.mean(all_data2, axis=0)



        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.84))
        # rectangular box plot
        bplot1 = axes[0].boxplot(all_data1,
                                 notch=True,  # notch shape
                                 vert=True,  # vertical box alignment
                                 patch_artist=True,  # fill with color
                                 showfliers=False,  # 异常值
                                 labels=labels)  # will be used to label x-ticks
        axes[0].set_title('Box plot on training data', fontsize=12)

        # notch shape box plot
        bplot2 = axes[1].boxplot(all_data2,
                                 notch=True,  # notch shape
                                 vert=True,  # vertical box alignment
                                 patch_artist=True,  # fill with color
                                 showfliers=False,
                                 labels=labels)  # will be used to label x-ticks
        axes[1].set_title('Box plot on test data', fontsize=12)
        # fill with colors
        colors = ['pink', 'lightblue', 'lightgreen', 'blue', 'aliceblue']
        for bplot in (bplot1, bplot2):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        # adding horizontal grid lines
        for ax in axes:
            ax.yaxis.grid(True)
            # ax.set_xlabel('Three separate samples')
        axes[0].set_ylabel(r'Error ($\%$)', fontsize=12)

        plt.savefig('box_plot_control.png', dpi=720)

        plt.show()



class Para2(object):
    def __init__(
            self,
            method,
            layer_num=6,
            units_num=256,
            activation_hidden=1,
            LR=0.0001,       # 学习率
            batch_size=256,  # 批次数量
            train=True,  # 训练的时候有探索
            tensorboard=True,
    ):
        self.method = method

        # train info
        self.trainInfo = Object()
        self.trainInfo.LR = LR
        self.trainInfo.tensorboard = tensorboard
        if train == 0:
            self.trainInfo.flag = True
        else:
            self.trainInfo.flag = False
        self.trainInfo.iteration = 0
        self.trainInfo.batch_size = batch_size

        # Obect info
        self.objectInfo = Object()
        self.objectInfo.r_dim = 3
        self.objectInfo.G_dim = 3

        # model info
        self.modelInfo = Object()
        self.modelInfo.layer_num = layer_num
        self.modelInfo.units_num = units_num
        self.modelInfo.activation_hidden = activation_hidden

        # data info
        ## data load path
        self.dataInfo = Object()
        d = path.dirname(__file__)  # 返回当前文件所在的目录
        parent_path = path.dirname(d)  # 获得d所在的目录,即d的父级目录
        data_path = parent_path + '/1 DataProcessing'
        self.dataInfo.train_data_path = data_path + '/train_data.npy'
        self.dataInfo.test_data_path = data_path + '/test_data.npy'
        ## model save
        model_path0 = os.path.join(sys.path[0], 'model')
        if not os.path.exists(model_path0):
            os.mkdir(model_path0)
        self.dataInfo.model_save_path = os.path.join(model_path0, 'data.chkp')
        ## tensorboard save
        self.dataInfo.log_save_path = './tensorboard/' + str(layer_num)+'/' + str(units_num) +'/'  + str(LR) +'/' + str(batch_size) +'/' \
                                    + str(activation_hidden) +'/' + self.method


class DataLoader2(object):
    def __init__(self, para):
        # train data 构建
        self.para =  para
        self.train_data = np.load(self.para.dataInfo.train_data_path)
        self.train_data_size = len(self.train_data[:, 0])
        self.test_data = np.load(self.para.dataInfo.test_data_path)
        self.test_data_size = len(self.test_data[:, 0])


        # train data 构建
        self.DataPreProcess()
        self.train_data = self.data_transform.transform(self.train_data)
        self.test_data = self.data_transform.transform(self.test_data)
        self.data_transform_scale_ = self.data_transform.scale_


    def get_batch(self):
        # get batch
        index = np.random.randint(0, np.shape(self.train_data)[0], self.para.trainInfo.batch_size)
        return self.train_data[index, :]

    def get_batch_test(self):
        # get batch
        index = np.random.randint(0, np.shape(self.test_data)[0], self.para.trainInfo.batch_size)
        return self.test_data[index, :]


    def DataPreProcess(self):
        # feature extraction
        self.data_transform = preprocessing.MinMaxScaler([-10, 10])
        self.data_transform.fit_transform(self.train_data)


class MLP2(object):
    def __init__(self, para):
        self.para = para
        self.data_loader = DataLoader2(para=self.para)
        model_graph = tf.Graph()
        self.sess_model = tf.Session(graph=model_graph)

        with model_graph.as_default():
            # placeholder
            self.r_input = tf.placeholder(tf.float32, [None, self.para.objectInfo.r_dim], 'state')
            self.G1_input = tf.placeholder(tf.float32, [None, 1], 'Gravity1')
            self.G2_input = tf.placeholder(tf.float32, [None, 1], 'Gravity2')
            self.G3_input = tf.placeholder(tf.float32, [None, 1], 'Gravity3')

            # build network
            if self.para.modelInfo.activation_hidden == 1:
                af_hidden = tf.nn.tanh
            elif self.para.modelInfo.activation_hidden == 2:
                af_hidden = tf.nn.relu
            else:
                af_hidden = None
                print("ActivationFuntion Setting Error")

            netname = "G1_net"
            with tf.variable_scope(netname):
                self.G1_pred = self._build_net(self.r_input, scope='eval', trainable=True,
                                               af_hidden=af_hidden, af_out=None, num_out=1)
                tf.summary.histogram(netname+'/eval', self.G1_pred)
                self.G1_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname+'/eval')
                self.G1_error = tf.subtract(self.G1_input, self.G1_pred, name=netname+'error')
                self.G1_MSE = tf.losses.mean_squared_error(labels=self.G1_input, predictions=self.G1_pred)
                self.G1_MAE = tf.reduce_mean(tf.abs(self.G1_error))
                self.G1_EMax = tf.norm(tf.abs(self.G1_error), ord=np.inf)
                self.G1_gradient = tf.gradients(self.G1_pred, self.r_input)
                tf.summary.scalar(netname+'MSE', self.G1_MSE)
                tf.summary.scalar(netname+'MAE', self.G1_MAE)
                tf.summary.scalar(netname+'Max', self.G1_EMax)
                self.G1_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.G1_MSE,
                                                                                        var_list=self.G1_params)

            netname = "G2_net"
            with tf.variable_scope(netname):
                self.G2_pred = self._build_net(self.r_input, scope='eval', trainable=True,
                                               af_hidden=af_hidden, af_out=None, num_out=1)
                tf.summary.histogram(netname + '/eval', self.G2_pred)
                self.G2_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.G2_error = tf.subtract(self.G2_input, self.G2_pred, name=netname + 'error')
                self.G2_MSE = tf.losses.mean_squared_error(labels=self.G2_input, predictions=self.G2_pred)
                self.G2_MAE = tf.reduce_mean(tf.abs(self.G2_error))
                self.G2_EMax = tf.norm(tf.abs(self.G2_error), ord=np.inf)
                self.G2_gradient = tf.gradients(self.G2_pred, self.r_input)
                tf.summary.scalar(netname + 'MSE', self.G2_MSE)
                tf.summary.scalar(netname + 'MAE', self.G2_MAE)
                tf.summary.scalar(netname + 'Max', self.G2_EMax)
                self.G2_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.G2_MSE,
                                                                                              var_list=self.G2_params)

            netname = "G3_net"
            with tf.variable_scope(netname):
                self.G3_pred = self._build_net(self.r_input, scope='eval', trainable=True,
                                               af_hidden=af_hidden, af_out=None, num_out=1)
                tf.summary.histogram(netname + '/eval', self.G3_pred)
                self.G3_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=netname + '/eval')
                self.G3_error = tf.subtract(self.G3_input, self.G3_pred, name=netname + 'error')
                self.G3_MSE = tf.losses.mean_squared_error(labels=self.G3_input, predictions=self.G3_pred)
                self.G3_MAE = tf.reduce_mean(tf.abs(self.G3_error))
                self.G3_EMax = tf.norm(tf.abs(self.G3_error), ord=np.inf)
                self.G3_gradient = tf.gradients(self.G3_pred, self.r_input)
                tf.summary.scalar(netname + 'MSE', self.G3_MSE)
                tf.summary.scalar(netname + 'MAE', self.G3_MAE)
                tf.summary.scalar(netname + 'Max', self.G3_EMax)
                self.G3_Train = tf.train.AdamOptimizer(self.para.trainInfo.LR).minimize(self.G3_MSE,
                                                                                        var_list=self.G3_params)
                self.actor_saver = tf.train.Saver()

        # sess initialization
        if self.para.trainInfo.flag:
            self.sess_model.run(tf.global_variables_initializer())
        else:
            with self.sess_model.as_default():
                with model_graph.as_default():
                    self.actor_saver.restore(self.sess_model, self.para.dataInfo.model_save_path)

        # tensorboard save
        if self.para.trainInfo.flag and self.para.trainInfo.tensorboard:
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.para.dataInfo.log_save_path, self.sess_model.graph)

    def model_save(self):
        self.actor_saver.save(self.sess_model, self.para.dataInfo.model_save_path)

    def _build_net(self, s, scope=None, trainable=True, af_hidden=tf.nn.relu, af_out=None, num_out=1):
        # 建立actor网络
        layer_num = self.para.modelInfo.layer_num
        units_num = self.para.modelInfo.units_num
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, units_num, activation=af_hidden, name='l0', trainable=trainable)
            for i in range(layer_num-2):
                net = tf.layers.dense(net, units_num, activation=af_hidden, name='l'+str(i+1), trainable=trainable)
            a = tf.layers.dense(net, num_out, activation=af_out, name='a', trainable=trainable)
            return a

    def predict(self, r):
        s = r[np.newaxis, :].copy()
        s = self.data_loader.data_transform.transform(np.hstack((s, s)))[:, 0:3]
        g1, g2, g3 = self.sess_model.run([self.G1_pred, self.G2_pred, self.G3_pred], {self.r_input: s})
        G = np.hstack((g1, g2, g3))
        G = self.data_loader.data_transform.inverse_transform(np.hstack((G, G)))
        return G[0, 3:6]

    def gradient_predict(self, r):
        s = r[np.newaxis, :].copy()
        s = self.data_loader.data_transform.transform(np.hstack((s, s)))[:, 0:3]
        g1_r_gradient, g2_r_gradient, g3_r_gradient \
            = self.sess_model.run([self.G1_gradient, self.G2_gradient, self.G3_gradient], {self.r_input: s})
        g1_r_gradient = g1_r_gradient[0] * self.data_loader.data_transform_scale_[0:3]/\
                        self.data_loader.data_transform_scale_[3]
        g2_r_gradient = g2_r_gradient[0] * self.data_loader.data_transform_scale_[0:3] / \
                        self.data_loader.data_transform_scale_[4]
        g3_r_gradient = g3_r_gradient[0] * self.data_loader.data_transform_scale_[0:3] / \
                        self.data_loader.data_transform_scale_[5]

        return g1_r_gradient[0], g2_r_gradient[0], g3_r_gradient[0]

    def gravity_predict(self, r):
        s = r[np.newaxis, :].copy()
        s = self.data_loader.data_transform.transform(np.hstack((s, s)))[:, 0:3]
        g1, g2, g3, g1_r_gradient, g2_r_gradient, g3_r_gradient \
            = self.sess_model.run([self.G1_pred, self.G2_pred, self.G3_pred,
                             self.G1_gradient, self.G2_gradient, self.G3_gradient], {self.r_input: s})
        G = np.hstack((g1, g2, g3))
        G = self.data_loader.data_transform.inverse_transform(np.hstack((G, G)))
        g1_r_gradient = g1_r_gradient[0] * self.data_loader.data_transform_scale_[0:3]/\
                        self.data_loader.data_transform_scale_[3]
        g2_r_gradient = g2_r_gradient[0] * self.data_loader.data_transform_scale_[0:3] / \
                        self.data_loader.data_transform_scale_[4]
        g3_r_gradient = g3_r_gradient[0] * self.data_loader.data_transform_scale_[0:3] / \
                        self.data_loader.data_transform_scale_[5]
        return G[0, 3:6], g1_r_gradient[0], g2_r_gradient[0], g3_r_gradient[0]


    def train(self):
        batch = self.data_loader.get_batch()
        r_batch = batch[:, 0:3]
        g1_batch = batch[:, 3:4]
        g2_batch = batch[:, 4:5]
        g3_batch = batch[:, 5:6]

        self.sess_model.run([self.G1_Train, self.G2_Train, self.G3_Train],
                      {self.r_input: r_batch, self.G1_input: g1_batch,
                       self.G2_input: g2_batch, self.G3_input: g3_batch})

        if self.para.trainInfo.tensorboard:
            if self.para.trainInfo.iteration % 100 == 0:
                result_merge = self.sess_model.run(self.merged, {self.r_input: r_batch,
                                            self.G1_input: g1_batch,
                                            self.G2_input: g2_batch,
                                            self.G3_input: g3_batch,
                                            })
                self.writer.add_summary(result_merge, self.para.trainInfo.iteration)
                print('Train', '-----------------', self.para.trainInfo.iteration)
                print('MAE', self.sess_model.run([self.G1_MAE, self.G2_MAE, self.G3_MAE],
                                           {self.r_input: r_batch,
                                            self.G1_input: g1_batch,
                                            self.G2_input: g2_batch,
                                            self.G3_input: g3_batch,
                                            }))
        self.para.trainInfo.iteration += 1

    def verify_train(self):
        batch = self.data_loader.get_batch()
        r_batch = batch[:, 0:3]
        g1_batch = batch[:, 3:4]
        g2_batch = batch[:, 4:5]
        g3_batch = batch[:, 5:6]

        print('Train MAE 1 2 3', self.sess_model.run([self.G1_MAE, self.G2_MAE, self.G3_MAE],
                                               {self.r_input: r_batch,
                                                self.G1_input: g1_batch,
                                                self.G2_input: g2_batch,
                                                self.G3_input: g3_batch,
                                                }))

    def verify_test(self):

        batch = self.data_loader.get_batch_test()
        r_batch = batch[:, 0:3]
        g1_batch = batch[:, 3:4]
        g2_batch = batch[:, 4:5]
        g3_batch = batch[:, 5:6]

        print('Test MAE 1 2 3', self.sess_model.run([self.G1_MAE, self.G2_MAE, self.G3_MAE],
                                               {self.r_input: r_batch,
                                                self.G1_input: g1_batch,
                                                self.G2_input: g2_batch,
                                                self.G3_input: g3_batch,
                                                }))

        G1_pred, G2_pred, G3_pred = self.sess_model.run([self.G1_pred, self.G2_pred, self.G3_pred],
                                                    {self.r_input: r_batch, self.G1_input: g1_batch,
                                                     self.G2_input: g2_batch, self.G3_input: g3_batch,})
        G_pred = np.hstack((G1_pred, G2_pred, G3_pred))
        print('------------Learn Gravity LearnCompare-----------------')
        print(np.hstack((batch[:, 3:6], G_pred)))
        print('------------Learn Gravity LearnError-----------------')
        G_error = batch[:, 3:6] - G_pred
        print(abs(G_error))
        print('------------Learn Gravity Origin Learn  %%%%% -----------------')
        print(np.linalg.norm(G_error, axis=1) / np.linalg.norm(batch[:, 3:6], axis=1) * 100)


        # Origin
        batch_origin = self.data_loader.data_transform.inverse_transform(batch)
        G_origin = batch_origin[:, 3:6]
        G_pred_origin = self.data_loader.data_transform.inverse_transform(np.hstack((G_pred, G_pred)))[:, 3:6]
        print('------------Origin Gravity LearnCompare-----------------')
        print(np.hstack((G_origin, G_pred_origin)))
        print('------------Origin Gravity LearnError-----------------')
        G_error = G_origin - G_pred_origin
        print(abs(G_error))
        print('------------Origin Gravity Origin Learn  %%%%% -----------------')
        print(np.linalg.norm(G_error, axis=1) / np.linalg.norm(G_origin, axis=1) * 100)























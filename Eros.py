
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import root, minimize




class Eros:

    def __init__(self, model, controller, random=False):
        self.t = None
        self.state = None
        self.model = model
        self.controller = controller
        self.random = random
        # 对象参数设置
        self.r_f_T = np.array([0, -3.7, 0])
        self.v_f_T = np.array([0., 0., 0.])
        self.r_0 = np.array([17.5, -30.3, 15])
        self.v_0 = np.array([0/1000, 0/1000, 0/1000])
        self.m0 = np.array([2000])

        self.Trust = 60 / 1000
        self.Isp = 400
        # self.tf = 3000
        self.g0 = 9.80665 / 1000
        self.epsilon_g = 0
        self.gravity_using = None


        # 仿真设置
        self.reset()
        self.state_dim, self.lambda_dim, self.control_dim = 7, 7, 3
        self.Tmax = 5000

        ## 目标行星 Eors
        self.omega = 3.311820212512959e-04
        self.G = 6.67e-11
        self.mass_Eros = 6.689e15


    def render(self):
        pass

    def reset(self, observation_input=None):
        self.t = 0
        if observation_input is None:
            if self.random is not True:
                self.state = np.hstack((self.r_0, self.v_0, self.m0))
            else:
               pass
        else:
            self.state = observation_input

        self.observation = self.state.copy()
        return self.observation

    def optimize_root(self, observation_input=None, op_variables=None):

        if observation_input is None:
            if op_variables is None:
                lambda_0 = np.random.rand(1)
                lambda_n = np.random.rand(self.lambda_dim-1) * 2 - 1
                lambda_m = np.random.rand(1)
                lambda_all = np.hstack([lambda_0, lambda_n, lambda_m])
                lambda_norm = lambda_all / np.linalg.norm(lambda_all)
                tf = np.random.rand(1)
                op_variables = np.hstack([lambda_norm, tf])
            else:
                pass
        else:
            self.reset(observation_input=observation_input)

        res = root(self.shoot_fun, op_variables, method='hybr', tol=1e-10, options={'factor': 1, 'band': None})
        # res = root(self.Fun_Shoot, costate_0)
        # samples, ceq, done, info = self.Fun_GetSamples(res.x)
        # print('hhhh', res.success)
        done = True if np.linalg.norm(res.fun) < 1e-6 else False
        return res, [], done, []

    def shoot_fun(self, op_variables):

        # print('op_variables', op_variables)

        self.lambda_0 = op_variables[0]
        lambda_n = op_variables[1:-1]
        lambda_all = op_variables[0:-1]
        self.tf = op_variables[-1] * self.Tmax

        if self.tf < 100:
            self.tf = 100
        elif self.tf > self.Tmax:
            self.tf  = self.Tmax


        # 微分方程
        X0 = np.hstack([self.state, lambda_n])
        # Xf = self.integrate_method(X0, method="RK45")
        # Xf = self.integrate_method(X0, method="LSODA")
        # Xf = self.integrate_method(X0, method="odeint")
        Xf = self.integrate_method(X0, method="RK45_Lynn")

        r = Xf[0:3]
        v = Xf[3:6]
        m = Xf[6]
        lambda_r = Xf[7:10]
        lambda_v = Xf[10:13]
        lambda_m = Xf[13]

        normLamV = np.linalg.norm(lambda_v)
        alpha = - lambda_v / normLamV

        rho = - self.Isp * self.g0 * normLamV / m - lambda_m

        if rho > 0:
            u = 0
        else:
            u = 1

        G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient = self.get_gravity(r)

        state_dot = np.zeros(shape=14)
        state_dot[3] = 2 * v[1] * self.omega + r[0] * self.omega * self.omega + G_pred[0] + self.Trust * u * alpha[
            0] / m
        state_dot[4] = -2 * v[0] * self.omega + r[1] * self.omega * self.omega + G_pred[1] + self.Trust * u * alpha[
            1] / m
        state_dot[5] = G_pred[2] + self.Trust * u * alpha[2] / m
        state_dot[6] = -self.Trust * u / self.Isp / self.g0



        # aa = Xf[10:13]
        # bb = state_dot[3:6]
        # H_end = aa.dot(bb) + self.lambda_0
        H_end = lambda_r.dot(v) + lambda_v.dot(state_dot[3:6]) + lambda_m * state_dot[6] + self.lambda_0/10000

        ceq = np.hstack((r-self.r_f_T, (v-self.v_f_T), lambda_m, np.linalg.norm(lambda_all)-1, H_end))
        # print('ceq', ceq)
        return ceq

    def integrate_method(self, X0, method="RK45"):

        if method == "RK45":
            X = integrate.RK45(self.motionEquation, t0=0, y0=X0, t_bound=self.tf, rtol=1e-10, atol=1e-10)
            while True:
                if X.t < self.tf:
                    X.step()
                else:
                    break
            Xf = X.y
        elif method == "LSODA":
            X = integrate.RK45(self.motionEquation, t0=0, y0=X0, t_bound=self.tf, rtol=1e-10, atol=1e-10)
            while True:
                if X.t < self.tf:
                    X.step()
                else:
                    break
            Xf = X.y
        elif method == "odeint":
            t = np.linspace(0, self.tf, 101)
            X = odeint(self.motionEquation2, X0, t, rtol=1e-10, atol=1e-10)
            Xf = X[-1, :]

        elif method == "RK45_Lynn":
            Xf = self.integrate_RK45(self.motionEquation, X0, num=200)
        else:
            Xf = 0
            print("intergrage error")

        return Xf

    def integrate_RK45(self, func, X0, num=101):

        delta_t = (self.tf-0)/(num-1)
        input = X0
        for step in range(num-1):
            k1 = self.motionEquation(0, input)
            k2 = self.motionEquation(0, input + delta_t * k1 / 2)
            k3 = self.motionEquation(0, input + delta_t * k2 / 2)
            k4 = self.motionEquation(0, input + delta_t * k3)
            input = input + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return input


    def motionEquation(self, t, input):

        lambda_0 = self.lambda_0

        r = input[0:3]
        v = input[3:6]
        m = input[6]
        lambda_r = input[7:10]
        lambda_v = input[10:13]
        lambda_m = input[13]

        normLamV = np.linalg.norm(lambda_v)
        alpha = - lambda_v / normLamV

        rho = - self.Isp * self.g0 * normLamV / m - lambda_m

        if rho > 0:
            u = 0
        else:
            u = 1

        G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient = self.get_gravity(r)

        state_dot = np.zeros(shape=14)
        state_dot[0:3] = v
        state_dot[3] = 2 * v[1] * self.omega + r[0] * self.omega * self.omega + G_pred[0] + self.Trust * u * alpha[0] / m
        state_dot[4] = -2 * v[0] * self.omega + r[1] * self.omega * self.omega + G_pred[1] + self.Trust * u * alpha[1] / m
        state_dot[5] = G_pred[2] + self.Trust * u * alpha[2] / m
        state_dot[6] = -self.Trust * u/self.Isp/self.g0
        state_dot[7] = - lambda_v[0]*self.omega*self.omega \
                       - lambda_v[0] * g1_r_gradient[0] - lambda_v[1] * g2_r_gradient[0] - lambda_v[2] * g3_r_gradient[0]
        state_dot[8] = - lambda_v[1] * self.omega * self.omega \
                       - lambda_v[0] * g1_r_gradient[1] - lambda_v[1] * g2_r_gradient[1] - lambda_v[2] * g3_r_gradient[1]
        state_dot[9] = - lambda_v[0] * g1_r_gradient[2] - lambda_v[1] * g2_r_gradient[2] - lambda_v[2] * g3_r_gradient[2]
        state_dot[10] = -lambda_r[0] + 2 * lambda_v[1]*self.omega
        state_dot[11] = -lambda_r[1] - 2 * lambda_v[0] * self.omega
        state_dot[12] = -lambda_r[2]
        state_dot[13] = - self.Trust * u * normLamV / m / m
        return state_dot

    def motionEquation2(self, input, t):
        return self.motionEquation(t, input)

    def get_gravity(self, r):

        gravity_using = self.gravity_using

        if gravity_using == 'Nomodel':
            G_pred = g1_r_gradient = g2_r_gradient = g3_r_gradient = [0, 0, 0]
        elif gravity_using == 'ParticleModel':
            epsilon_g = self.epsilon_g
            r_norm = np.linalg.norm(r)
            x = r[0]
            y = r[1]
            z = r[2]
            mu = self.G * self.mass_Eros* 1e-9
            G_pred = - mu / r_norm**3 * r
            g1_x = - mu / r_norm**3 + 3 * mu * x * x / r_norm**5
            g1_y =  3 * mu * x * y / r_norm ** 5
            g1_z =  3 * mu * x * z / r_norm ** 5
            g1_r_gradient = np.hstack((g1_x, g1_y, g1_z))
            g2_x =  3 * mu * y * x / r_norm ** 5
            g2_y = - mu / r_norm ** 3 + 3 * mu * y * y / r_norm ** 5
            g2_z = 3 * mu * y * z / r_norm ** 5
            g2_r_gradient = np.hstack((g2_x, g2_y, g2_z))
            g3_x = 3 * mu * z * x / r_norm ** 5
            g3_y = 3 * mu * z * y / r_norm ** 5
            g3_z = - mu / r_norm ** 3 + 3 * mu * z * z / r_norm ** 5
            g3_r_gradient = np.hstack((g3_x, g3_y, g3_z))
            G_pred *= epsilon_g
            g1_r_gradient *= epsilon_g
            g2_r_gradient *= epsilon_g
            g3_r_gradient *= epsilon_g
        elif gravity_using == 'NetworkModel':
            epsilon_g = self.epsilon_g
            if 60 > np.linalg.norm(r) > 3:
                # G_pred = epsilon_g * self.model.predict(np.array(r[0:3]))
                # g1_r_gradient, g2_r_gradient, g3_r_gradient = self.model.gradient_predict(np.array(r[0:3]))
                G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient = self.model.gravity_predict(np.array(r[0:3]))
                g1_r_gradient *= epsilon_g
                g2_r_gradient *= epsilon_g
                g3_r_gradient *= epsilon_g
            else:
                G_pred = g1_r_gradient = g2_r_gradient = g3_r_gradient = [0, 0, 0]
        else:
            print('gravity_using setting error')

        deviation = self.deviation
        G_pred = G_pred * deviation
        g1_r_gradient = g1_r_gradient * deviation
        g2_r_gradient = g2_r_gradient * deviation
        g3_r_gradient = g3_r_gradient * deviation
        return G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient


    def get_trajectory(self, op_variables):

        self.lambda_0 = op_variables[0]
        lambda_n = op_variables[1:-1]
        self.tf = op_variables[-1]

        X_tra = np.empty([0, 6])
        X_acc = np.empty([0, 6])
        X_accgrad = np.empty([0, 12])
        X_u1 = np.empty([0, 1])

        X0 = np.hstack([self.state, lambda_n])
        G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient = self.get_gravity(X0[0:3])
        X_tra = np.vstack((X_tra, X0[0:6]))
        X_acc = np.vstack((X_acc, np.hstack((X0[0:3], G_pred))))
        X_accgrad = np.vstack((X_accgrad, np.hstack((X0[0:3], g1_r_gradient, g2_r_gradient, g3_r_gradient))))

        num=200
        delta_t = (self.tf - 0) / (num - 1)

        input = X0
        for step in range(num - 1):

            r = input[0:3]
            v = input[3:6]
            m = input[6]
            lambda_r = input[7:10]
            lambda_v = input[10:13]
            lambda_m = input[13]

            normLamV = np.linalg.norm(lambda_v)
            alpha = - lambda_v / normLamV

            rho = - self.Isp * self.g0 * normLamV / m - lambda_m

            if rho > 0:
                u = 0
            else:
                u = 1

            k1 = self.motionEquation(0, input)
            k2 = self.motionEquation(0, input + delta_t * k1 / 2)
            k3 = self.motionEquation(0, input + delta_t * k2 / 2)
            k4 = self.motionEquation(0, input + delta_t * k3)
            input = input + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            X_tra = np.vstack((X_tra, input[0:6]))
            G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient = self.get_gravity(np.array(input[0:3]))
            X_acc = np.vstack((X_acc, np.hstack((input[0:3], G_pred))))
            X_accgrad = np.vstack((X_accgrad, np.hstack((input[0:3], g1_r_gradient, g2_r_gradient, g3_r_gradient))))
            X_u1 = np.vstack((X_u1, u))


        Tra = {}
        Tra["X_tra"] = X_tra
        Tra["X_acc"] = X_acc
        Tra["X_accgrad"] = X_accgrad
        Tra["X_u1"] = X_u1
        Tra["mass_consump"] = self.m0 - input[6]
        Tra["ceq"] = np.hstack((input[0:3] - self.r_f_T, (input[3:6] - self.v_f_T) * 1000, lambda_m))

        return Tra

    def get_samples(self, op_variables):

        self.lambda_0 = op_variables[0]
        lambda_n = op_variables[1:-1]
        self.tf = op_variables[-1] * self.Tmax

        X0 = np.hstack([self.state, lambda_n])


        sample_tra = np.empty([0, 7])
        sample_lambda = np.empty([0, 8])
        sample_control = np.empty([0, 4])
        sample_mass_consume = np.empty([0, 1])
        sample_time_consume = np.empty([0, 1])


        num = 200
        delta_t = (self.tf - 0) / (num - 1)

        input = X0
        for step in range(num - 1):

            r = input[0:3]
            v = input[3:6]
            m = input[6]
            lambda_r = input[7:10]
            lambda_v = input[10:13]
            lambda_m = input[13]

            sample_tra = np.vstack((sample_tra, input[0:7]))
            sample_lambda = np.vstack((sample_lambda, np.hstack((self.lambda_0, input[7:14]))))
            sample_mass_consume = np.vstack((sample_mass_consume, m))
            sample_time_consume = np.vstack((sample_time_consume, delta_t*step))

            normLamV = np.linalg.norm(lambda_v)
            alpha = - lambda_v / normLamV

            rho = - self.Isp * self.g0 * normLamV / m - lambda_m

            if rho > 0:
                u = 0
            else:
                u = 1
            sample_control = np.vstack((sample_control, np.hstack((u, alpha))))

            k1 = self.motionEquation(0, input)
            k2 = self.motionEquation(0, input + delta_t * k1 / 2)
            k3 = self.motionEquation(0, input + delta_t * k2 / 2)
            k4 = self.motionEquation(0, input + delta_t * k3)
            input = input + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6



        sample_mass_consume = sample_mass_consume - input[6]
        sample_time_consume = self.tf - sample_time_consume

        samples = {}
        samples["sample_tra"] = sample_tra
        samples["sample_lambda"] = sample_lambda
        samples["sample_control"] = sample_control
        samples["sample_mass_consume"] = sample_mass_consume
        samples["sample_time_consume"] = sample_time_consume
        samples["ceq"] = np.hstack((input[0:3] - self.r_f_T, (input[3:6] - self.v_f_T), lambda_m))

        return samples


    def get_driven(self, end_flag=1):

        X0 = np.hstack([self.state])
        sample_tra = np.empty([0, 7])
        sample_control = np.empty([0, 4])
        sample_mass_consume = np.empty([0, 1])
        sample_time_consume = np.empty([0, 1])

        if end_flag == 0:
            num = 200
            delta_t = (self.tf - 0) / (num - 1)
            input = X0
            for step in range(num - 1):
                state7 = input[0:7]
                alpha = self.controller.control_predict(state7)[0]
                sample_tra = np.vstack((sample_tra, input[0:7]))
                sample_mass_consume = np.vstack((sample_mass_consume, input[6]))
                sample_time_consume = np.vstack((sample_time_consume, delta_t*step))
                u = 1
                sample_control = np.vstack((sample_control, np.hstack((u, alpha))))

                k1 = self.motionEquation_u(0, input)
                k2 = self.motionEquation_u(0, input + delta_t * k1 / 2)
                k3 = self.motionEquation_u(0, input + delta_t * k2 / 2)
                k4 = self.motionEquation_u(0, input + delta_t * k3)
                input = input + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            state7 = input[0:7]
            alpha = self.controller.control_predict(state7)[0]
            sample_tra = np.vstack((sample_tra, input[0:7]))
            sample_mass_consume = np.vstack((sample_mass_consume, input[6]))
            sample_time_consume = np.vstack((sample_time_consume, delta_t * (num-1)))
            u = 1
            sample_control = np.vstack((sample_control, np.hstack((u, alpha))))

        elif end_flag == 1:
            delta_t = 1
            input = X0
            # print(input)
            for step in range(100000):
                # print(input)
                state7 = input[0:7]
                distance1 = np.linalg.norm(input[0:3] - self.r_f_T)
                alpha = self.controller.control_predict(state7)[0]
                sample_tra = np.vstack((sample_tra, input[0:7]))
                sample_mass_consume = np.vstack((sample_mass_consume, input[6]))
                sample_time_consume = np.vstack((sample_time_consume, delta_t * step))
                u = 1
                sample_control = np.vstack((sample_control, np.hstack((u, alpha))))

                k1 = self.motionEquation_u(0, input)
                k2 = self.motionEquation_u(0, input + delta_t * k1 / 2)
                k3 = self.motionEquation_u(0, input + delta_t * k2 / 2)
                k4 = self.motionEquation_u(0, input + delta_t * k3)
                input = input + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                distance2 = np.linalg.norm(input[0:3] - self.r_f_T)

                print(distance1, distance2)

                if distance1 < 5 and distance2 > distance1:
                    break

        state7 = input[0:7]
        alpha = self.controller.control_predict(state7)[0]
        sample_tra = np.vstack((sample_tra, input[0:7]))
        sample_mass_consume = np.vstack((sample_mass_consume, input[6]))
        sample_time_consume = np.vstack((sample_time_consume, delta_t * (num - 1)))
        u = 1
        sample_control = np.vstack((sample_control, np.hstack((u, alpha))))


        sample_mass_consume = sample_mass_consume - input[6]
        sample_time_consume = self.tf - sample_time_consume

        samples = {}
        samples["sample_tra"] = sample_tra
        samples["sample_control"] = sample_control
        samples["sample_mass_consume"] = sample_mass_consume
        samples["sample_time_consume"] = sample_time_consume
        samples["ceq"] = np.hstack((input[0:3] - self.r_f_T, (input[3:6] - self.v_f_T)))

        return samples

    def motionEquation_u(self, t, input):

        state7 = input[0:7]
        r = input[0:3]
        v = input[3:6]
        m = input[6]

        alpha = self.controller.control_predict(state7)[0]
        u = 1

        G_pred, g1_r_gradient, g2_r_gradient, g3_r_gradient = self.get_gravity(r)

        state_dot = np.zeros(shape=7)
        state_dot[0:3] = v
        state_dot[3] = 2 * v[1] * self.omega + r[0] * self.omega * self.omega + G_pred[0] + self.Trust * u * alpha[0] / m
        state_dot[4] = -2 * v[0] * self.omega + r[1] * self.omega * self.omega + G_pred[1] + self.Trust * u * alpha[1] / m
        state_dot[5] = G_pred[2] + self.Trust * u * alpha[2] / m
        state_dot[6] = -self.Trust * u/self.Isp/self.g0
        return state_dot


























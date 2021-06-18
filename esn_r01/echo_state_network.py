import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from copy import deepcopy
from sys_data import *
from linear_regression import *


class EchoState(object):
    """description of class"""

    def __init__(self, system: SysData, esn_dict):
        # configurações ESN
        self.tau_de_serie = esn_dict['tau e de']
        self.neuronios = esn_dict['numero de neuronios']
        self.raio = esn_dict['raio do reservatorio']
        self.sparcity = esn_dict['esparcidade']
        self.seed_esn = esn_dict['seed dos pesos']
        self.ruido_reser = esn_dict['ruido: reservatorio']
        self.ruido_saida = esn_dict['ruido: saida']
        self.faixa_pesos_entrada = esn_dict['faixa de pesos: entrada']
        self.faixa_pesos_saida = esn_dict['faixa de pesos: saida']
        self.faixa_pesos_bias = esn_dict['faixa de pesos: bias']

        self.const_reg = esn_dict['const reg']

        self.saida_estimada = []
        self.norm = []
        self.rmse = []
        self.rmse_train = []
        self.pesos_camada_saida = []
        self.buffer_series = []

        # configurações sistema
        self.system = system
        vetor = []
        for stepDe in range(self.tau_de_serie[1]):
            vetor.append(-1 - stepDe * self.tau_de_serie[0])
        self.vetor_takens_serie = np.sort(np.asarray(vetor))
        print(self.vetor_takens_serie)

        # ################################################################
        # ## Matriz W: esparça
        np.random.seed(self.seed_esn)
        self.pesos_reservatorio = 2 * np.random.rand(self.neuronios, self.neuronios) - 1
        condition = np.absolute(self.pesos_reservatorio) > self.sparcity
        self.pesos_reservatorio[condition] = 0
        print('Esparcidade do reservatório:', np.count_nonzero(self.pesos_reservatorio),
              np.count_nonzero(self.pesos_reservatorio) / self.neuronios ** 2)
        # ### Normalização
        w, v = np.linalg.eig(self.pesos_reservatorio)
        w = w[np.argsort(-w)]

        self.pesos_reservatorio = self.raio * (1 / np.absolute(w[0])) * self.pesos_reservatorio
        w, v = np.linalg.eig(self.pesos_reservatorio)
        w = w[np.argsort(-w)]
        print('Raio espectral do reservatório:', np.absolute(w[0]))

        # ## Pesos aleatório bias
        np.random.seed(self.seed_esn + 1)
        self.pesos_bias = self.faixa_pesos_bias * (2 * np.random.rand(self.neuronios) - 1)

        # ## Pesos aleatório da entrada
        np.random.seed(self.seed_esn + 2)
        self.pesos_entrada = self.faixa_pesos_entrada * (2 * np.random.rand(self.neuronios, self.system.num_series) - 1)

        # ################################################################
        # Estado Inicial
        self.x_n1 = np.zeros(self.neuronios)

        # Treinamento
        self.echo_rls()

    def echo_rls(self):
        list_saida_camada_oculta = []
        list_saida = []
        x_n1 = self.x_n1.copy()
        list_saida_camada_oculta.append(x_n1)

        self.saida_estimada = np.zeros([1, self.system.num_series])

        self.buffer_series = np.zeros([int(-self.vetor_takens_serie[0]), self.system.num_series])
        for step in range(self.system.num_series):
            self.buffer_saida[:, step] = (self.buffer_saida[:, step] - self.system.mean_data[step]) / \
                                         self.system.std_data[step]

        # ESN TRAINING
        for step in range(self.system.train_point):
            # print(step)
            aux_saida_conc = np.asarray([])
            for step_conc in range(self.system.num_series):
                aux_saida_conc = np.concatenate(
                    [aux_saida_conc, self.buffer_saida[self.vetorTakens_saida, step_conc]])

            if self.saida_no_reser:
                auxx_saida = np.inner(self.pesos_saida, aux_saida_conc)  # + self.noise_saida[:, step]
            else:
                auxx_saida = np.zeros(self.neuronios)

            aux_entrada_conc = np.asarray([])
            for step_conc in range(self.system.num_in):
                aux_entrada_conc = np.concatenate([aux_entrada_conc,
                                                   self.buffer_entrada[self.vetorTakens_entrada, step_conc]])

            auxx_entrada = np.inner(self.pesos_entrada, aux_entrada_conc)

            auxx_reservatorio = np.inner(self.pesos_reservatorio, list_saida_camada_oculta[-1])  #

            list_saida_camada_oculta.append(np.tanh(
                auxx_entrada + auxx_saida + auxx_reservatorio - self.pesos_bias))  # + self.noise_res[:, step]

            if self.saida_no_W_out:
                aux_readout_saida = aux_saida_conc  # + self.noise_saida[:, step]
            else:
                aux_readout_saida = np.asarray([])

            list_saida.append(np.concatenate([list_saida_camada_oculta[-1], [-1], aux_entrada_conc, aux_readout_saida]))

            if step == 0:
                num_pesos = list_saida[-1].shape[0]
                rls = LrRLS(num_pesos, self.system.num_out, self.const_reg)
                rls = LrLMS3(num_pesos, self.system.num_out)
                # rls = LrLmsBF(num_pesos, self.system.num_out)
                self.pesos_camada_saida = np.zeros([num_pesos, self.system.num_out])  # iniciar os pesos

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            self.saida_estimada = np.concatenate([self.saida_estimada, aux_saida_est])

            # RLS

            self.pesos_camada_saida = rls.update(self.system.sys_out[step:step + 1, :], aux_saida_est, list_saida[-1])

            self.buffer_entrada = np.concatenate([self.buffer_entrada, self.system.sys_in[step + 1:step + 2, :]])
            self.buffer_saida = np.concatenate([self.buffer_saida, self.system.sys_out[step:step + 1, :]])

        # print('teste')
        for step1 in range(self.system.final_point - self.system.train_point):
            step = self.system.train_point + step1
            aux_saida_conc = np.asarray([])
            for step_conc in range(self.system.num_out):
                aux_saida_conc = np.concatenate(
                    [aux_saida_conc, self.buffer_saida[self.vetorTakens_saida, step_conc]])
            if self.saida_no_reser:
                auxx_saida = np.inner(self.pesos_saida, aux_saida_conc)
            else:
                auxx_saida = np.zeros(self.neuronios)

            aux_entrada_conc = np.asarray([])
            for step_conc in range(self.system.num_in):
                aux_entrada_conc = np.concatenate([aux_entrada_conc,
                                                   self.buffer_entrada[self.vetorTakens_entrada, step_conc]])

            auxx_entrada = np.inner(self.pesos_entrada, aux_entrada_conc)

            auxx_reservatorio = np.inner(self.pesos_reservatorio, list_saida_camada_oculta[-1])  #

            list_saida_camada_oculta.append(np.tanh(auxx_entrada + auxx_saida + auxx_reservatorio - self.pesos_bias))

            if self.saida_no_W_out:
                aux_readout_saida = aux_saida_conc
            else:
                aux_readout_saida = np.asarray([])

            list_saida.append(np.concatenate([list_saida_camada_oculta[-1], [-1], aux_entrada_conc, aux_readout_saida]))

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            # print('teste:', self.saida_estimada.shape)
            cond = aux_saida_est > 2
            aux_saida_est[cond] = 2
            cond = aux_saida_est < -2
            aux_saida_est[cond] = -2
            self.saida_estimada = np.concatenate([self.saida_estimada, aux_saida_est])

            self.buffer_saida = np.concatenate([self.buffer_saida, aux_saida_est])

            if step == self.system.final_point - 1:
                break
            self.buffer_entrada = np.concatenate([self.buffer_entrada, self.system.sys_in[step + 1:step + 2, :]])

        # Valor Estimado
        '''
        plt.figure()
        plt.plot(self.saida_estimada[1:, :], marker='.', linewidth=0.6)
        plt.plot(self.sistema.saida_treino, marker='.', linewidth=0.6)
        #plt.show()
        '''
        self.saida_estimada = self.saida_estimada[1:, :]
        self.norm = []
        self.rmse = []
        self.rmse_train = []

        for step in range(self.system.num_out):
            self.saida_estimada[:, step] = self.saida_estimada[:, step] * self.system.std_sys_out[step] \
                                           + self.system.mean_sys_out[step]

            self.norm.append(np.linalg.norm(self.pesos_camada_saida[:, step], 2))
            # print(self.pesos_camada_saida[:, step].shape)
            self.rmse.append(np.sqrt(np.mean((self.saida_estimada[self.system.train_point:, step]
                                              - self.system.sys_target[self.system.train_point:, step]) ** 2)))
            self.rmse_train.append(np.sqrt(np.mean(
                (self.saida_estimada[int(self.system.train_point / 2):self.system.train_point, step]
                 - self.system.sys_target[int(self.system.train_point / 2):self.system.train_point, step]) ** 2)))

        print('norma do vetor de pesos:', self.norm)
        print("RMSE:", self.rmse)
        print("RMSE Trenio:", self.rmse_train)

        # '''
        for step in range(self.system.num_out):
            plt.figure()
            plt.plot(self.saida_estimada[:, step], marker='.', linewidth=0.6)
            plt.plot(self.system.sys_out[:self.system.train_point, step] *
                     self.system.std_sys_out[step] + self.system.mean_sys_out[step], marker='.', linewidth=0.6)
            plt.plot(self.system.sys_target[:, step], marker='.', linestyle='--', linewidth=0.6)
            plt.axvline(x=self.system.train_point - 1, color='k', linestyle='--', linewidth=0.6)
            plt.axvline(x=int(self.system.train_point/2), color='b', linestyle='--', linewidth=0.6)
            plt.title("Predição RLS")
            plt.legend(["Estimada", "Corropida", "Desejada"], loc=2)
            # plt.show()

        plt.figure()
        estados = np.asarray(list_saida_camada_oculta)
        print(estados.shape)
        plt.plot(estados[self.system.train_point:, :5], marker='.', linewidth=0.6)

        plt.figure()
        plt.plot(np.asarray(rls.erro)[:, 0])

        # '''

    def echo_MRLS(self, outlier_detect=True):
        list_saida_camada_oculta = []
        list_saida = []
        saida_estimada = []
        x_n1 = self.x_n1
        list_saida_camada_oculta.append(x_n1)

        self.saida_estimada = np.zeros([1, self.sistema.num_saida])

        self.buffer_saida = np.zeros([int(-self.vetorTakens_saida[0]), self.sistema.num_saida])
        for step in range(self.sistema.num_saida):
            self.buffer_saida[:, step] = (self.buffer_saida[:, step] - self.sistema.mean_saida_treino[step]) / \
                                         self.sistema.std_saida_treino[step]

        self.buffer_entrada = np.zeros([int(-self.vetorTakens_entrada[0]), self.sistema.num_entrada])
        for step in range(self.sistema.num_entrada):
            self.buffer_entrada[:, step] = (self.buffer_entrada[:, step] - self.sistema.mean_entrada_treino[step]) / \
                                           self.sistema.std_entrada_treino[step]

        self.buffer_entrada = np.concatenate([self.buffer_entrada, self.sistema.entrada[0:1, :]])
        entrou = 0
        print('treino')
        for step in range(self.sistema.ponto_treino):
            # print(step)
            noise_saida = self.ruido_saida * (2 * np.random.rand(self.sistema.num_saida
                                                                 * self.vetorTakens_saida.shape[0]) - 1)
            aux_saida_conc = np.asarray([])
            for step_conc in range(self.sistema.num_saida):
                aux_saida_conc = np.concatenate(
                    [aux_saida_conc, self.buffer_saida[self.vetorTakens_saida, step_conc]])

            if self.saida_no_reser:
                auxx_saida = np.inner(self.pesos_saida, aux_saida_conc + noise_saida)
            else:
                auxx_saida = np.zeros(self.neuronios)

            aux_entrada_conc = np.asarray([])
            for step_conc in range(self.sistema.num_entrada):
                aux_entrada_conc = np.concatenate([aux_entrada_conc,
                                                   self.buffer_entrada[self.vetorTakens_entrada, step_conc]])

            auxx_entrada = np.inner(self.pesos_entrada, aux_entrada_conc)

            auxx_reservatorio = np.inner(self.pesos_reservatorio, list_saida_camada_oculta[-1])  #

            noise_res = self.ruido_reser * (2 * np.random.rand(self.neuronios) - 1)

            list_saida_camada_oculta.append(np.tanh(
                auxx_entrada + auxx_saida + auxx_reservatorio - self.pesos_bias + noise_res))

            if self.saida_no_W_out:
                aux_readout_saida = aux_saida_conc + noise_saida
            else:
                aux_readout_saida = np.asarray([])

            list_saida.append(np.concatenate([[-1], list_saida_camada_oculta[-1], aux_entrada_conc, aux_readout_saida]))

            if step == 0:
                num_pesos = list_saida[-1].shape[0]
                # print('numero de pesos',num_pesos)
                P = []
                W = []
                k = []
                limiar = []
                alfa = []
                erro_post = []
                outlier = []
                for step_out in range(self.sistema.num_saida):
                    P.append(1e+4 * np.eye(num_pesos))
                    W.append(np.zeros([num_pesos, 1]))
                    k.append(np.zeros([num_pesos, 1]))
                    limiar.append(100)
                    alfa.append(1)
                    erro_post.append([])
                    outlier.append(False)

                self.pesos_camada_saida = np.zeros([num_pesos, self.sistema.num_saida])  # iniciar os pesos

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            self.saida_estimada = np.concatenate([self.saida_estimada, aux_saida_est])

            # RLS
            erro = self.sistema.saida_treino[step:step + 1, :] - aux_saida_est
            # print(erro)
            aux_erro = np.abs(erro)
            # print(aux_erro)
            saida_estimada_post = np.zeros([1, self.sistema.num_saida])
            for step_rlm in range(self.sistema.num_saida):
                if aux_erro[0, step_rlm] / limiar[step_rlm] < 1:  # or epoca < 100:
                    q_erro = 1
                    outlier[step_rlm] = False
                else:
                    q_erro = 0  # limiar[step_rlm] / aux_erro[0, step_rlm]
                    if not outlier[step_rlm]:
                        outlier[step_rlm] = True

                k[step_rlm] = q_erro * np.inner(P[step_rlm], list_saida[-1]) / (
                        alfa[step_rlm] + q_erro * np.inner(list_saida[-1], np.inner(P[step_rlm], list_saida[-1])))
                W[step_rlm][:, 0] = W[step_rlm][:, 0] + k[step_rlm] * erro[0, step_rlm]
                P[step_rlm] = (1 / alfa[step_rlm]) * (P[step_rlm]
                                                      - np.inner(np.outer(k[step_rlm], list_saida[-1]), P[step_rlm]))
                # print(W[step_rlm].shape)
                aux_W = np.reshape(W[step_rlm], (W[step_rlm].shape[0], 1)).T
                # print('erro post',erro_post)
                erro_post[step_rlm].append(aux_erro[0, step_rlm] ** 2)
                # print(len(erro_post[step_rlm]))
                det = np.max([len(erro_post[step_rlm]) - 1, 1])
                limiar[step_rlm] = 2.576 * np.sqrt(1.483 * (1 + 5 / det) * np.median(erro_post[step_rlm]))

            self.pesos_camada_saida = np.asarray(W)[:, :, 0].T

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            if outlier_detect and min(outlier):
                entrou = entrou + 1

                saida_estimada_post = aux_saida_est
            else:
                saida_estimada_post = self.sistema.saida_treino[step:step + 1:, :]

            self.buffer_entrada = np.concatenate([self.buffer_entrada, self.sistema.entrada[step + 1:step + 2, :]])
            self.buffer_saida = np.concatenate([self.buffer_saida, saida_estimada_post])
        print('entrou', entrou)
        print('teste')
        for step1 in range(self.sistema.ponto_final - self.sistema.ponto_treino):
            step = self.sistema.ponto_treino + step1
            # print(step)
            aux_saida_conc = np.asarray([])
            for step_conc in range(self.sistema.num_saida):
                aux_saida_conc = np.concatenate(
                    [aux_saida_conc, self.buffer_saida[self.vetorTakens_saida, step_conc]])
            if self.saida_no_reser:
                auxx_saida = np.inner(self.pesos_saida, aux_saida_conc)
            else:
                auxx_saida = np.zeros(self.neuronios)

            aux_entrada_conc = np.asarray([])
            for step_conc in range(self.sistema.num_entrada):
                aux_entrada_conc = np.concatenate([aux_entrada_conc,
                                                   self.buffer_entrada[self.vetorTakens_entrada, step_conc]])

            auxx_entrada = np.inner(self.pesos_entrada, aux_entrada_conc)

            auxx_reservatorio = np.inner(self.pesos_reservatorio, list_saida_camada_oculta[-1])  #

            list_saida_camada_oculta.append(np.tanh(auxx_entrada + auxx_saida + auxx_reservatorio - self.pesos_bias))

            if self.saida_no_W_out:
                aux_readout_saida = aux_saida_conc + noise_saida
            else:
                aux_readout_saida = np.asarray([])

            list_saida.append(np.concatenate([[-1], list_saida_camada_oculta[-1], aux_entrada_conc, aux_readout_saida]))

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            # print('teste:', self.saida_estimada.shape)
            cond = aux_saida_est > 2
            aux_saida_est[cond] = 2
            cond = aux_saida_est < -2
            aux_saida_est[cond] = -2
            self.saida_estimada = np.concatenate([self.saida_estimada, aux_saida_est])

            self.buffer_saida = np.concatenate([self.buffer_saida, aux_saida_est])

            if step == self.sistema.ponto_final - 1:
                break
            self.buffer_entrada = np.concatenate([self.buffer_entrada, self.sistema.entrada[step + 1:step + 2, :]])

        # Valor Estimado
        '''
        plt.figure()
        plt.plot(self.saida_estimada[1:, :], marker='.', linewidth=0.6)
        plt.plot(self.sistema.saida_treino, marker='.', linewidth=0.6)
        #plt.show()
        '''

        self.saida_estimada = self.saida_estimada[1:, :]
        self.norm = []
        self.RMSE = []
        for step in range(self.sistema.num_saida):
            self.saida_estimada[:, step] = self.saida_estimada[:, step] * self.sistema.std_saida_treino[step] \
                                           + self.sistema.mean_saida_treino[step]

            self.norm.append(np.linalg.norm(self.pesos_camada_saida[:, step], 2))
            # print(self.pesos_camada_saida[:, step].shape)
            self.RMSE.append(np.sqrt(np.mean((self.saida_estimada[self.sistema.ponto_treino:, step]
                                              - self.sistema.saida_desejada[self.sistema.ponto_treino:, step]) ** 2)))
        print('norma do vetor de pesos:', self.norm)
        print("RMSE:", self.RMSE)

        # '''
        for step in range(self.sistema.num_saida):
            plt.figure()
            plt.plot(self.saida_estimada[:, step], marker='.', linewidth=0.6)
            plt.plot(self.sistema.saida_treino[:self.sistema.ponto_treino, step] *
                     self.sistema.std_saida_treino[step] + self.sistema.mean_saida_treino[step], marker='.',
                     linewidth=0.6)
            plt.plot(self.sistema.saida_desejada[:, step], marker='.', linestyle='--', linewidth=0.6)
            plt.axvline(x=self.sistema.ponto_treino - 1, color='k', linestyle='--', linewidth=0.6)
            plt.title("Predição RLM")
            plt.legend(["Estimada", "Corropida", "Desejada"], loc=2)
            # plt.show()
        # '''

    def echo_RLS_one(self, saida_esc=0):
        list_saida_camada_oculta = []
        list_saida = []
        saida_estimada = []
        x_n1 = self.x_n1
        list_saida_camada_oculta.append(x_n1)

        self.saida_estimada = np.zeros([1, self.sistema.num_saida])

        self.buffer_saida = np.zeros([int(-self.vetorTakens_saida[0]), self.sistema.num_saida])
        for step in range(self.sistema.num_saida):
            self.buffer_saida[:, step] = (self.buffer_saida[:, step] - self.sistema.mean_saida_treino[step]) / \
                                         self.sistema.std_saida_treino[step]

        self.buffer_entrada = np.zeros([int(-self.vetorTakens_entrada[0]), self.sistema.num_entrada])
        for step in range(self.sistema.num_entrada):
            self.buffer_entrada[:, step] = (self.buffer_entrada[:, step] - self.sistema.mean_entrada_treino[step]) / \
                                           self.sistema.std_entrada_treino[step]

        self.buffer_entrada = np.concatenate([self.buffer_entrada, self.sistema.entrada[0:1, :]])

        # print('treino')
        for step in range(self.sistema.ponto_treino):
            # print(step)
            noise_saida = self.ruido_saida * (2 * np.random.rand(self.vetorTakens_saida.shape[0]) - 1)
            aux_saida_conc = self.buffer_saida[self.vetorTakens_saida, saida_esc]

            if self.saida_no_reser:
                auxx_saida = np.inner(self.pesos_saida[:, saida_esc: saida_esc + 1], aux_saida_conc + noise_saida)
            else:
                auxx_saida = np.zeros(self.neuronios)

            aux_entrada_conc = np.asarray([])
            for step_conc in range(self.sistema.num_entrada):
                aux_entrada_conc = np.concatenate([aux_entrada_conc,
                                                   self.buffer_entrada[self.vetorTakens_entrada, step_conc]])

            auxx_entrada = np.inner(self.pesos_entrada, aux_entrada_conc)

            auxx_reservatorio = np.inner(self.pesos_reservatorio, list_saida_camada_oculta[-1])  #

            noise_res = self.ruido_reser * (2 * np.random.rand(self.neuronios) - 1)
            # print(auxx_entrada.shape)
            # print(auxx_saida.shape)
            list_saida_camada_oculta.append(np.tanh(
                auxx_entrada + auxx_saida + auxx_reservatorio - self.pesos_bias + noise_res))

            if self.saida_no_W_out:
                aux_readout_saida = aux_saida_conc + noise_saida
            else:
                aux_readout_saida = np.asarray([])

            list_saida.append(np.concatenate([[-1], list_saida_camada_oculta[-1], aux_entrada_conc, aux_readout_saida]))

            if step == 0:
                num_pesos = list_saida[-1].shape[0]
                # print('numero de pesos',num_pesos)
                P = 1e+4 * np.eye(num_pesos)
                alfa = 1
                W = np.zeros([num_pesos, self.sistema.num_saida])
                self.pesos_camada_saida = np.zeros([num_pesos, self.sistema.num_saida])  # iniciar os pesos

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            self.saida_estimada = np.concatenate([self.saida_estimada, aux_saida_est])

            # RLS
            erro = self.sistema.saida_treino[step:step + 1, :] - aux_saida_est
            # print(erro, erro.shape)
            k = np.inner(P, list_saida[-1]) / (alfa + np.inner(list_saida[-1], np.inner(P, list_saida[-1])))
            k = np.reshape(k, (k.shape[0], 1))
            # print(k.shape)
            W = W + np.inner(k, erro.T)
            # print(W.shape)
            P = (1 / alfa) * (P - np.inner(np.outer(k, list_saida[-1]), P))
            self.pesos_camada_saida = np.reshape(W, (W.shape[0], self.sistema.num_saida))

            self.buffer_entrada = np.concatenate([self.buffer_entrada, self.sistema.entrada[step + 1:step + 2, :]])
            self.buffer_saida = np.concatenate([self.buffer_saida, self.sistema.saida_treino[step:step + 1, :]])

        # print('teste')
        for step1 in range(self.sistema.ponto_final - self.sistema.ponto_treino):
            step = self.sistema.ponto_treino + step1
            aux_saida_conc = self.buffer_saida[self.vetorTakens_saida, saida_esc]

            if self.saida_no_reser:
                auxx_saida = np.inner(self.pesos_saida[:, saida_esc: saida_esc + 1], aux_saida_conc)
            else:
                auxx_saida = np.zeros(self.neuronios)

            aux_entrada_conc = np.asarray([])
            for step_conc in range(self.sistema.num_entrada):
                aux_entrada_conc = np.concatenate([aux_entrada_conc,
                                                   self.buffer_entrada[self.vetorTakens_entrada, step_conc]])

            auxx_entrada = np.inner(self.pesos_entrada, aux_entrada_conc)

            auxx_reservatorio = np.inner(self.pesos_reservatorio, list_saida_camada_oculta[-1])  #

            list_saida_camada_oculta.append(np.tanh(auxx_entrada + auxx_saida + auxx_reservatorio - self.pesos_bias))

            if self.saida_no_W_out:
                aux_readout_saida = aux_saida_conc + noise_saida
            else:
                aux_readout_saida = np.asarray([])

            list_saida.append(np.concatenate([[-1], list_saida_camada_oculta[-1], aux_entrada_conc, aux_readout_saida]))

            aux_saida_est = \
                np.inner(self.pesos_camada_saida.T, np.reshape(list_saida[-1], (list_saida[-1].shape[0], 1)).T).T
            # print('teste:', self.saida_estimada.shape)
            cond = aux_saida_est > 2
            aux_saida_est[cond] = 2
            cond = aux_saida_est < -2
            aux_saida_est[cond] = -2
            self.saida_estimada = np.concatenate([self.saida_estimada, aux_saida_est])

            self.buffer_saida = np.concatenate([self.buffer_saida, aux_saida_est])

            if step == self.sistema.ponto_final - 1:
                break
            self.buffer_entrada = np.concatenate([self.buffer_entrada, self.sistema.entrada[step + 1:step + 2, :]])

        # Valor Estimado
        '''
        plt.figure()
        plt.plot(self.saida_estimada[1:, :], marker='.', linewidth=0.6)
        plt.plot(self.sistema.saida_treino, marker='.', linewidth=0.6)
        #plt.show()
        '''
        self.saida_estimada = self.saida_estimada[1:, :]
        self.norm = []
        self.RMSE = []
        for step in range(self.sistema.num_saida):
            self.saida_estimada[:, step] = self.saida_estimada[:, step] * self.sistema.std_saida_treino[step] \
                                           + self.sistema.mean_saida_treino[step]

            self.norm.append(np.linalg.norm(self.pesos_camada_saida[:, step], 2))
            # print(self.pesos_camada_saida[:, step].shape)
            self.RMSE.append(np.sqrt(np.mean((self.saida_estimada[self.sistema.ponto_treino:, step]
                                              - self.sistema.saida_desejada[self.sistema.ponto_treino:, step]) ** 2)))
        print('norma do vetor de pesos:', self.norm)
        print("RMSE:", self.RMSE)

        # '''
        for step in range(self.sistema.num_saida):
            plt.figure()
            plt.plot(self.saida_estimada[:, step], marker='.', linewidth=0.6)
            plt.plot(self.sistema.saida_treino[:self.sistema.ponto_treino, step] *
                     self.sistema.std_saida_treino[step] + self.sistema.mean_saida_treino[step], marker='.',
                     linewidth=0.6)
            plt.plot(self.sistema.saida_desejada[:, step], marker='.', linestyle='--', linewidth=0.6)
            plt.axvline(x=self.sistema.ponto_treino - 1, color='k', linestyle='--', linewidth=0.6)
            plt.title("Predição RLS")
            plt.legend(["Estimada", "Corropida", "Desejada"], loc=2)
            # plt.show()
        # '''

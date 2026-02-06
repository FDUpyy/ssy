import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from model.eval_methods import *


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A
def matrix_poly(matrix, d):
    x = torch.eye(d).double().cuda()+ torch.div(matrix.cuda(), d).cuda()
    return torch.matrix_power(x, d)
def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        # 自上次模型在验证集上损失降低之后等待的时间，此处设置为7
        self.verbose = verbose
        # 当为False时，运行的时候将不显示详细信息
        self.counter = 0
        # 计数器，当其值超过patience时候，使用class EarlyStopping
        self.best_score = None
        # 记录模型评估的最好分数
        self.best_score2 = None
        self.early_stop = False
        # 决定模型要不要early stop，为True则停
        self.val_loss_min = np.Inf
        # 模型评估损失函数的最小值，默认为正无穷(np.Inf)
        self.val_loss2_min = np.Inf
        self.delta = delta
        # 表示模型损失函数改进的最小值，当超过这个值时候表示模型有所改进
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        # __call__()使得类的实例对象可以像类一样调用（实例调用的是__call__()）
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            # save_checkpoint()保存最佳模型
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    # object可以继承类的很多高级对象
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        # self.__dict__.update()来使某个字典中的键值变成可以使用的变量
        # **表示字典类型的不定长参数
        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        # self.data_path, self.batch_size, self.win_size, self.dataset应该是输入config的一些字典

        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)
        # thre_loader是什么？
        self.build_model()
        # 直接调用build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        # 用于计算重构损失

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, t_out=self.output_t, delays=self.delays, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # model.parameters()多用于优化器的初始化，输出一些权重/偏置参数
        # model.named_parameters()输出层参数名字和参数值

        if torch.cuda.is_available():
            self.model.cuda()
            # model.cuda()表示将模型的参数放到gpu上，另外我们还要将该模型用到的输入数据放到同一gpu上

    def vali(self, vali_loader):
    # 用于调超参
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            # i = batch索引
            # input_data = (B, window_size, input_dim)
            # _是labels，验证集和训练集用不到

            input = input_data.float().to(self.device)

            output, series, prior, sigmas, causal = self.model(input[:, :, :int(input.shape[2]/2)], input[:, :, int(input.shape[2]/2):])
            #output, series, prior, sigmas, causal = self.model(input[:, :, int(input.shape[2]/2):], input[:, :, :int(input.shape[2]/2)])

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            input = input[:, :, int(input.shape[2] / 2):]
            rec_loss = self.criterion(output, input)

            causal_batch = causal.mean(dim=1)
            # causal_batch = (B D D)

            # 因果不变loss
            for c in range(2):
                causal = causal.mean(dim=0)
            # causal = (D D)
            causal_b = causal.repeat(causal_batch.shape[0], 1, 1)
            # causal_b = (B D D)
            causal_loss = 1000 * torch.mean(my_kl_loss(causal_batch, causal_b) + my_kl_loss(causal_b, causal_batch))

            # 因果矩阵稀疏loss
            sparse_loss = self.tau_A * torch.sum(torch.abs(causal))

            # Constraint A
            h_A = 200 * _h_A(causal, len(causal))

            """
            # 仅rec_loss：
            loss_1.append(rec_loss.item())
            
            # 仅series_loss：
            loss_1.append((-series_loss).item())

            # 仅sparse_loss：
            loss_1.append(sparse_loss.item())

            # 仅causal_loss：
            loss_1.append(causal_loss.item())
            
            # 仅A_loss：
            loss_1.append((self.lambda_A * h_A + 0.5 * h_A * h_A).item())
            """
            
            loss_1.append((rec_loss - series_loss + causal_loss + self.lambda_A * h_A + 0.5 * h_A * h_A).item())
            loss_2.append((rec_loss + prior_loss + causal_loss + self.lambda_A * h_A + 0.5 * h_A * h_A).item())


        return np.average(loss_1), np.average(loss_2)
        #return np.average(loss_1)

    def train(self):

        print("======================TRAIN MODE======================")
        temperature = 50
        time_now = time.time()
        # 用于获取当前时间的时间戳
        path = self.model_save_path
        # config字典里的
        criterion = nn.MSELoss(reduce=False)
        if not os.path.exists(path):
            os.makedirs(path)
            # os.makedirs()用于递归创建目录,如果所要创建的目录已经存在，那么python将抛出OSError
            # 如果makedirs()参数指定所要创建的目标目录中的某一个节点路径不存在，则makedirs()会自动创建该节点路径
            # path在class EarlyStopping中有使用
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        # len(self.train_loader) = batch数量
        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            rec_loss_list = []
            series_loss_list = []
            prior_loss_list = []
            sparse_loss_list = []
            causal_loss_list = []
            A_loss_list = []

            epoch_time = time.time()
            self.model.train()
            # main()调用solver后，调用build_model()里的self.model
            # self.model.train()启用batch normalization和dropout
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.train_loader):
            # enumerate(self.train_loader)得到batch_size个__getitem__(i)返回的数据列表
            # __len__()返回batch的个数

                self.optimizer.zero_grad()
                iter_count += 1

                input = input_data.float().to(self.device)

                #print(input[:, :, :int(input.shape[2]/2)].shape)

                output, series, prior, sigmas, causal = self.model(input[:, :, :int(input.shape[2] / 2)], input[:, :, int(input.shape[2] / 2):])
                #output, series, prior, sigmas, causal = self.model(input, input)
                # causal = (B H D D)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                series_score = 0.0
                prior_score = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    prior_loss += (torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)),
                                   series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(),
                                   (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)))))
                series_loss = series_loss / len(prior)
                #print("series shape:", series_loss.shape)
                prior_loss = prior_loss / len(prior)

                input = input[:, :, int(input.shape[2] / 2):]
                rec_loss = self.criterion(output, input)

                causal_batch = causal.mean(dim=1)
                #print(causal_batch.shape)
                # causal_batch = (B D D)

                # 因果不变loss
                for c in range(2):
                    causal = causal.mean(dim=0)
                # causal = (D D)
                causal_b = causal.repeat(causal_batch.shape[0], 1, 1)

                # causal_b = (B D D)
                causal_loss = 1000 * torch.mean(my_kl_loss(causal_batch, causal_b) + my_kl_loss(causal_b, causal_batch))

                # 因果矩阵稀疏loss
                sparse_loss = self.tau_A * torch.sum(torch.abs(causal))
                # Constraint A
                h_A = 200 * _h_A(causal, len(causal))

                rec_loss_list.append(rec_loss.item())
                series_loss_list.append(series_loss.item())
                prior_loss_list.append(prior_loss.item())
                sparse_loss_list.append(sparse_loss.item())
                causal_loss_list.append(causal_loss.item())
                A_loss_list.append((self.lambda_A * h_A + 0.5 * h_A * h_A).item())
                """
                # 仅rec_loss：
                loss1_list.append(rec_loss.item())
                loss1 = rec_loss
                # 仅series_loss：
                loss1_list.append((-series_loss).item())
                loss1 = -series_loss
                # 仅sparse_loss：
                loss1_list.append(sparse_loss.item())
                loss1 = sparse_loss
                # 仅causal_loss：
                loss1_list.append(causal_loss.item())
                loss1 = causal_loss
                # 仅A_loss：
                loss1_list.append((self.lambda_A * h_A + 0.5 * h_A * h_A).item())
                loss1 = self.lambda_A * h_A + 0.5 * h_A * h_A
                """

                # 全loss：
                loss1_list.append((rec_loss - series_loss + causal_loss + self.lambda_A * h_A + 0.5 * h_A * h_A).item())
                loss1 = rec_loss - series_loss + causal_loss + self.lambda_A * h_A + 0.5 * h_A * h_A
                loss2 = rec_loss + prior_loss + causal_loss + self.lambda_A * h_A + 0.5 * h_A * h_A

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                # 保存上一次计算的梯度不被释放
                loss2.backward()
                # 释放计算图
                self.optimizer.step()
                # 参数更新，使得损失函数最小化
                """
                # Min strategy
                loss1.backward()
                self.optimizer.step()
                # 参数更新，使得损失函数最小化
                """

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            reconstruction_loss = np.average(rec_loss_list)
            series_loss_final = np.average(series_loss_list)
            prior_loss_final = np.average(prior_loss_list)
            sparse_loss_final = np.average(sparse_loss_list)
            causal_loss_final = np.average(causal_loss_list)
            A_loss_final = np.average(A_loss_list)

            # Minimax strategy
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)
            # self.vali是在train()内部，所以不需要保存参数，自动使用训练好的参数计算vali数据集的loss

            # Min strategy
            # vali_loss1= self.vali(self.test_loader)
            # vali_loss2= np.Inf

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Rec Loss: {4:.7f} Series Loss: {5:.7f} Prior Loss: {6:.7f} Sparse Loss: {7:.7f} Causal Loss: {8:.7f} A Loss: {9:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss1, reconstruction_loss, series_loss_final, prior_loss_final, sparse_loss_final, causal_loss_final, A_loss_final))
            early_stopping(vali_loss1, vali_loss2, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            # break后跳出循环，不再执行adjust_learning_rate

            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            # calculate train score:

        causal_original = causal

        return causal_original



    def test(self):
        causal_original = self.train()
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        # load_state_dict()加载模型参数
        # os.path.join()路径拼接函数：连接两个或更多的路径名组件
        # pth文件包含了模型在训练过程中学到的权重参数
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)
        # reduce = False则返回向量形式的loss（每个样本的loss）


        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):

            input = input_data.float().to(self.device)

            output, series, prior, sigmas, causal = self.model(input[:, :, :int(input.shape[2]/2)], input[:, :, int(input.shape[2]/2):])
            causal_train = causal_original
            # causal_original = (D D)

            input = input[:, :, int(input.shape[2] / 2):]
            loss = torch.mean(criterion(input, output), dim=-1)
            # loss = (B L)

            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                # 为什么分开写？不懂
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    # series_loss 返回（B L）
                    # Rescaled P, 将其转换为离散分布
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature


            causal = causal.mean(dim=1)
            # causal = (B D D)

            causal_train = causal_train.repeat(len(causal), 1).view(len(causal), len(causal_original), len(causal_original))
            # causal_train = (B D D)

            causal_loss = 100 * (my_kl_loss(causal, causal_train) + my_kl_loss(causal_train, causal)).repeat(series_loss.shape[1], 1).transpose(1, 0) * temperature

            # causal_loss = (B L)

            """
            # 仅rec
            cri = loss
            # 仅series
            cri = torch.softmax(-metric, dim=-1)
            
            # 仅causal
            cri = torch.softmax(causal_loss, dim=-1)
            
            #只计算窗口内最后一个data point的异常分数
            cri =0.0
            for w in range(metric.shape[1]):
                cri += (1 / abs(w - metric.shape[1])) * metric[:,w] * loss[:,w]
            # cri即AnomalyScore
            cri = cri / w
            """
            metric = causal_loss - series_loss - prior_loss
            # metric = (B L)

            cri = torch.softmax(metric, dim=-1) * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            # attens_energy = (i B)
            # attens_energy = (i B L)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # np.concatenate(attens_energy, axis=0) = (i*B  L)
        # attens_energy = (i*B*L)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            output, series, prior, sigmas, causal = self.model(input[:, :, :int(input.shape[2]/2)], input[:, :, int(input.shape[2]/2):])
            causal_train = causal_original

            input = input[:, :, int(input.shape[2] / 2):]
            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            causal = causal.mean(dim=1)
            # causal = (B D D)
            causal_train = causal_train.repeat(len(causal), 1).view(len(causal), len(causal_original), len(causal_original))

            # causal_original = (B D D)
            causal_loss = 100 * (my_kl_loss(causal, causal_train) + my_kl_loss(causal_train, causal)).repeat(series_loss.shape[1], 1).transpose(1, 0) * temperature
            # causal_loss = (B L)

            # Metric
            metric = causal_loss - series_loss - prior_loss
            # metric = (B L)

            cri = torch.softmax(metric, dim=-1) * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # combined_energy = (2*i*B*L)

        thresh_count = np.count_nonzero(combined_energy)
        print("count:", thresh_count)
        #thresh = np.percentile(combined_energy, 100 * (1- thresh_count / combined_energy.shape[0]))
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        # percentile()：求combined_energy中第100 - self.anomaly_ratio百分位的值,从小到大排列

        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)

            output, series, prior, sigmas, causal = self.model(input[:, :, :int(input.shape[2]/2)], input[:, :, int(input.shape[2]/2):])
            causal_train = causal_original
            input = input[:, :, int(input.shape[2] / 2):]
            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature


            causal = causal.mean(dim=1)
            # causal = (B D D)
            causal_train = causal_train.repeat(len(causal), 1).view(len(causal), len(causal_original), len(causal_original))
            # causal_original = (B D D)
            causal_loss = 100 * (my_kl_loss(causal, causal_train) + my_kl_loss(causal_train, causal)).repeat(series_loss.shape[1], 1).transpose(1, 0) * temperature
            # causal_loss = (B L)

            # Metric
            metric = causal_loss - series_loss - prior_loss
            # metric = (B L)

            cri = torch.softmax(metric, dim=-1) * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_thresh_count = np.count_nonzero(test_energy)
        print("test_count:", test_thresh_count)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        # 计算测试数据标签，符合1不符合0
        gt = test_labels.astype(int)
        # 真实标签

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)


        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        """
        # get best f1
        t, th = bf_search(test_energy, test_labels, start=0.03, end=0.08, step_num=int(abs(0.08 - 0.03) / 0.00001), display_freq=1000)
                              
        # output the results
           
        print({
            'best-f1': t[0],
            'precision': t[1],
            'recall': t[2],
            'TP': t[3],
            'TN': t[4],
            'FP': t[5],
            'FN': t[6],
            'latency': t[-1],
            'threshold': th
        })  
        """

        # get pot results
        pot_result = pot_eval(train_energy, test_energy, test_labels, level_t=0.01 * self.anomaly_ratio)
        print({
                'pot-f1': pot_result[0],
                'pot-precision': pot_result[1],
                'pot-recall': pot_result[2],
                'pot-TP': pot_result[3],
                'pot-TN': pot_result[4],
                'pot-FP': pot_result[5],
                'pot-FN': pot_result[6],
                'pot-threshold': pot_result[7],
                'pot-latency': pot_result[-1]
        })

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score

        





        

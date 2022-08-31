import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
import scipy
from sklearn.metrics import roc_auc_score
import pickle
import os.path
from scipy import io
import sys

use_cuda = torch.cuda.is_available()

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=True):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class MTGNN(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(MTGNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = E2EBlock(1, 8, example)
        self.E2N = torch.nn.Conv2d(8,1, (1,self.d))
        self.fc1 = torch.nn.Linear(1,64)
        self.fc2 = torch.nn.Linear(64,27)

        self.fc3_Fi = torch.nn.Linear(27,3)

        self.fc3_Fo = torch.nn.Linear(27, 3)

        self.fc3_T = torch.nn.Linear(27, 3)

        self.fc3_L = torch.nn.Linear(27, 3)

    def forward(self, x):

        out = self.e2econv1(x)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = out.view(out.size(2), out.size(1))
        out = self.fc1(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = self.fc2(out)
        out = F.leaky_relu(out,negative_slope=0.33)

        out_Fi = F.leaky_relu(self.fc3_Fi(out), negative_slope=0.33)

        out_Fo = F.leaky_relu(self.fc3_Fo(out), negative_slope=0.33)

        out_T = F.leaky_relu(self.fc3_T(out), negative_slope=0.33)

        out_L = F.leaky_relu(self.fc3_L(out), negative_slope=0.33)
        return out_Fi, out_Fo, out_T, out_L


import torch.utils.data.dataset

for test in range(10):
    test_index = test+1
    lr = 0.005
    nbepochs = 104
    BATCH_SIZE = 1
    class_0 = 0.12
    class_M = 1.27
    class_L = 2.02
    class_2 = 0.3

    trainset = data_train(index=test_index,fold=10)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    testset = data_test(index=test_index)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    net = MTGNN(trainset.X)
    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])

    momentum = 0.9
    wd = 0.00005  ## Decay for L2 regularization

    def init_weights_he(m):
        print(m)
        if type(m) == torch.nn.Linear:
            fan_in = net.dense1.in_features
            he_lim = np.sqrt(6) / fan_in
            m.weight.data.uniform_(-he_lim, he_lim)
            print(m.weight)

    class_weight_M = torch.FloatTensor([class_0, class_M, class_2])
    criterion1 = torch.nn.CrossEntropyLoss(weight=class_weight_M)
    class_weight_L = torch.FloatTensor([class_0, class_L, class_2])
    criterion2 = torch.nn.CrossEntropyLoss(weight=class_weight_L)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)

    def train(epoch):
        net.train()
        for batch_idx, (X, L, Fi, Fo, T) in enumerate(trainloader):


            if use_cuda:
                X,L,Fi,Fo,T = X.cuda(), L.cuda(), Fi.cuda(), Fo.cuda(), T.cuda()
            optimizer.zero_grad()
            X, L, Fi, Fo, T = Variable(X), Variable(L), Variable(Fi), Variable(Fo), Variable(T)
            out_Fi, out_Fo, out_T, out_L = net(X)
            L = L.view(L.size(0) * L.size(1), 1)
            L = np.squeeze(L)
            L = Variable(L)
            Fi = Fi.view(Fi.size(0) * Fi.size(1), 1)
            Fi = np.squeeze(Fi)
            Fi = Variable(Fi)
            Fo = Fo.view(Fo.size(0) * Fo.size(1), 1)
            Fo = np.squeeze(Fo)
            Fo = Variable(Fo)
            T = T.view(T.size(0) * T.size(1), 1)
            T = np.squeeze(T)
            T = Variable(T)

            loss1 = criterion2(out_L,L)
            if Fi[0] == 6:
                loss2 = 0
            else:
                loss2 = criterion1(out_Fi,Fi)
            if Fo[0] == 6:
                loss3 = 0
            else:
                loss3 = criterion1(out_Fo,Fo)
            if T[0] == 6:
                loss4 = 0
            else:
                loss4 = criterion1(out_T,T)
            loss_total = loss1 + loss2 + loss3 + loss4
            loss_total.backward()
            optimizer.step()
            print(loss_total)
        return

    def test():
        net.eval()
        test_loss = 0
        running_loss = 0.0

        tot_acc_L = []
        tot_L = []
        tot_auc_L = []
        tot_acc_Fi = []
        tot_Fi = []
        tot_auc_Fi = []
        tot_acc_Fo = []
        tot_Fo = []
        tot_auc_Fo = []
        tot_acc_T = []
        tot_T = []
        tot_auc_T = []

        for batch_idx, (X, L, Fi, Fo, T) in enumerate(testloader):

            if use_cuda:
                X,L,Fi,Fo,T = X.cuda(), L.cuda(), Fi.cuda(), Fo.cuda(), T.cuda()

            with torch.no_grad():

                X, L, Fi, Fo, T = Variable(X), Variable(L), Variable(Fi), Variable(Fo), Variable(T)
                out_Fi, out_Fo, out_T, out_L = net(X)
                L = np.squeeze(L)
                L = L.data.numpy()
                Fi = np.squeeze(Fi)
                Fi = Fi.data.numpy()
                Fo = np.squeeze(Fo)
                Fo = Fo.data.numpy()
                T = np.squeeze(T)
                T = T.data.numpy()

                _, pred_L = torch.max(out_L, 1)
                _, pred_Fi = torch.max(out_Fi, 1)
                _, pred_Fo = torch.max(out_Fo, 1)
                _, pred_T = torch.max(out_T, 1)

                pred_L = pred_L.cpu()
                pred_Fi = pred_Fi.cpu()
                pred_Fo = pred_Fo.cpu()
                pred_T = pred_T.cpu()

                out_L = out_L.cpu()
                out_L_numpy = out_L.data.numpy()
                out_Fi = out_Fi.cpu()
                out_Fi_numpy = out_Fi.data.numpy()
                out_Fo = out_Fo.cpu()
                out_Fo_numpy = out_Fo.data.numpy()
                out_T = out_T.cpu()
                out_T_numpy = out_T.data.numpy()

                #if statement for checking existence here before AUC's
                tumor_index = np.where(L == 2)[0]
                L[tumor_index] = 0
                auc_L = roc_auc_score(L, out_L_numpy[:, 1])
                a = np.asarray(L)
                b = np.asarray(pred_L)
                L_parcels = np.count_nonzero(a == 1)
                TP = 0
                correct = 0
                for i in range(384):
                    if a[i] == b[i]:
                        correct = correct + 1
                    if a[i] == 1 and b[i] == 1:
                        TP = TP + 1
                acc_L = float(correct) / 384
                class_L = float(TP) / float(L_parcels)


                if Fi[0] == 6:
                    auc_Fi = -1
                    acc_Fi = -1
                    class_Fi = -1
                else:
                    Fi[tumor_index] = 0
                    auc_Fi = roc_auc_score(Fi, out_Fi_numpy[:, 1])
                    a = np.asarray(Fi)
                    b = np.asarray(pred_Fi)
                    Fi_parcels = np.count_nonzero(a == 1)
                    TP = 0
                    correct = 0
                    for i in range(384):
                        if a[i] == b[i]:
                            correct = correct + 1
                        if a[i] == 1 and b[i] == 1:
                            TP = TP + 1
                    acc_Fi = float(correct) / 384
                    class_Fi = float(TP) / float(Fi_parcels)
                if Fo[0] == 6:
                    auc_Fo = -1
                    acc_Fo = -1
                    class_Fo = -1
                else:
                    Fo[tumor_index] = 0
                    auc_Fo = roc_auc_score(Fo, out_Fo_numpy[:, 1])
                    a = np.asarray(Fo)
                    b = np.asarray(pred_Fo)
                    Fo_parcels = np.count_nonzero(a == 1)
                    TP = 0
                    correct = 0
                    for i in range(384):
                        if a[i] == b[i]:
                            correct = correct + 1
                        if a[i] == 1 and b[i] == 1:
                            TP = TP + 1
                    acc_Fo = float(correct) / 384
                    class_Fo = float(TP) / float(Fo_parcels)
                if T[0] == 6:
                    auc_T = -1
                    acc_T = -1
                    class_T = -1
                else:
                    T[tumor_index] = 0
                    auc_T = roc_auc_score(T, out_T_numpy[:, 1])
                    a = np.asarray(T)
                    b = np.asarray(pred_T)
                    T_parcels = np.count_nonzero(a == 1)
                    TP = 0
                    correct = 0
                    for i in range(384):
                        if a[i] == b[i]:
                            correct = correct + 1
                        if a[i] == 1 and b[i] == 1:
                            TP = TP + 1
                    acc_T = float(correct) / 384
                    class_T = float(TP) / float(T_parcels)


                tot_auc_L.append(auc_L)
                tot_auc_Fi.append(auc_Fi)
                tot_auc_Fo.append(auc_Fo)
                tot_auc_T.append(auc_T)
                tot_acc_L.append(acc_L)
                tot_L.append(class_L)
                tot_acc_Fi.append(acc_Fi)
                tot_Fi.append(class_Fi)
                tot_acc_Fo.append(acc_Fo)
                tot_Fo.append(class_Fo)
                tot_acc_T.append(acc_T)
                tot_T.append(class_T)

        return tot_auc_L,tot_acc_L,tot_L, tot_auc_Fi,tot_acc_Fi,tot_Fi, tot_auc_Fo,tot_acc_Fo,tot_Fo, tot_auc_T,tot_acc_T,tot_T

    for epoch in range(nbepochs):

        train(epoch)

    auc_L, overall_L, class_L, auc_Fi, overall_Fi, class_Fi, auc_Fo, overall_Fo, class_Fo, auc_T, overall_T, class_T = test()



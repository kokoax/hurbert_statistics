#! coding: utf-8
import sys
import random
import math
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets

class Count:
    def __init__(self):
        self.matrix = {}
        self.n_keys = []
        self.n = 0
        self.M = 0
        self.contingency = {}
        self.data_flg = 0
        self.all_data_sets = None
        self.data_sets = self.getDataSets()
        for data in self.data_sets:
            print(data["name"])
        self.show_data()
        # print(self.matrix)
        self.n = self.get_n()
        self.set_contingency()
        _sum = 0
        self.m1 = self.get_m1()
        self.m2 = self.get_m2()
        self.m3 = self.get_m3()
        self.m4 = self.get_m4()
        self.M = self.get_M()
        print("gamma: ", self.get_gamma())
        print(self.contingency['0000'])

    def data_load(filename):
        f = open("input/".join(filename))

        line = f.readline()
        ret = []
        while line:
            ret.append(map(lambda val:float(val), re.split(r'[,:]',line)))
            line = f.readline()
        self.ncol = ret.size()
        self.nrow = ret[0].size()

    def getDataSets(self):
        if self.data_flg == 0:
            data_sets = datasets.load_iris()
        elif self.data_flg == 1:
            data_sets = datasets.load_digits()
        elif self.data_flg == 2:
            return data_load("haberman.data")

        self.all_data_sets = data_sets
        self.nrow, self.ncol = data_sets.data.shape
        tmp = 2
        return [
                {
                    'name':data_sets.target[i+50*tmp],
                    'data':data_sets.data[i+50*tmp],
                    'cluster':-1
                } for i in range(int(self.nrow/3))
        ]
    # 特徴xaにおいてxixjが同一クラスタかどうか
    def Ia(self, i, j):
        if self.nomalize(self.data_sets[i]["data"][0]) == self.nomalize(self.data_sets[j]["data"][0]):
            return 1
        else:
            return 0
    def Ib(self, i, j):
        if self.nomalize(self.data_sets[i]["data"][1]) == self.nomalize(self.data_sets[j]["data"][1]):
            return 1
        else:
            return 0
    def Ic(self, i, j):
        if self.nomalize(self.data_sets[i]["data"][2]) == self.nomalize(self.data_sets[j]["data"][2]):
            return 1
        else:
            return 0
    def Id(self, i, j):
        if self.nomalize(self.data_sets[i]["data"][3]) == self.nomalize(self.data_sets[j]["data"][3]):
            return 1
        else:
            return 0

    def set_contingency(self):
        self.contingency = {}
        print(self.n)
        for i in range(self.n-1):
            for j in range(self.n-i-1):
                key = "".join([str(self.Ia(i,i+j+1)),str(self.Ib(i,i+j+1)),str(self.Ic(i,i+j+1)),str(self.Id(i,i+j+1))])
                if key in self.contingency:
                    self.contingency[key] += 1
                else:
                    self.contingency[key] = 1

    def get_m1(self):
        count = 0
        count += self.contingency["1000"]
        count += self.contingency["1001"]
        count += self.contingency["1010"]
        count += self.contingency["1011"]
        count += self.contingency["1100"]
        count += self.contingency["1101"]
        count += self.contingency["1110"]
        count += self.contingency["1111"]
        return count

    def get_m2(self):
        count = 0
        count += self.contingency["0100"]
        count += self.contingency["0101"]
        count += self.contingency["0110"]
        count += self.contingency["0111"]
        count += self.contingency["1100"]
        count += self.contingency["1101"]
        count += self.contingency["1110"]
        count += self.contingency["1111"]
        return count

    def get_m3(self):
        count = 0
        count += self.contingency["0010"]
        count += self.contingency["0011"]
        count += self.contingency["0110"]
        count += self.contingency["0111"]
        count += self.contingency["1010"]
        count += self.contingency["1011"]
        count += self.contingency["1110"]
        count += self.contingency["1111"]
        return count

    def get_m4(self):
        count = 0
        count += self.contingency["0001"]
        count += self.contingency["0011"]
        count += self.contingency["0101"]
        count += self.contingency["0111"]
        count += self.contingency["1001"]
        count += self.contingency["1011"]
        count += self.contingency["1101"]
        count += self.contingency["1111"]
        return count

    def get_M(self):
        return (self.n*(self.n-1)) / 2

    def get_gamma(self):
        print(self.M, self.m1, self.m2, self.m3, self.m4)
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3
        m4 = self.m4
        M = self.M
        return (self.contingency["1111"]/M - (m1/M)*(m2/M)*(m3/M)*(m4/M)) / math.sqrt((m1/M-(m1/M)**2)*(m2/M-(m2/M)**2)*(m3/M-(m3/M)**2)*(m4/M-(m4/M)**2))
        # return ((self.M*self.contingency["1111"])-ip_m) / (math.sqrt(ip_m*ip_minus_m))

    def get_n(self):
        n = 0
        for x1_key in self.n_keys[0]:
            for x2_key in self.n_keys[1]:
                for x3_key in self.n_keys[2]:
                    for x4_key in self.n_keys[3]:
                        key = "".join([x1_key,x2_key,x3_key,x4_key])
                        n += self.matrix[key]
        return n

    def nomalize(self, vec):
        under= vec - int(vec)
        if_five = 0
        if vec - int(vec) >= 0.5:
            if_five = 0.5
        return float(int(vec)) + if_five

    def show_data(self):
        count = [{} for i in range(4)]
        for data in self.data_sets:
            for key in data.keys():
                if key in "data":
                    i = 0
                    generated_key = []
                    for vec in data["data"]:
                        conv = self.nomalize(vec)
                        generated_key.append(str(conv))
                        if str(conv) in count[i]:
                            count[i][str(conv)] += 1
                        else:
                            count[i][str(conv)] = 1
                        i += 1
                    if "".join(generated_key) in self.matrix:
                        self.matrix["".join(generated_key)] += 1
                    else:
                        self.matrix["".join(generated_key)] = 1
        sort_count = []
        for counted in count:
            sort_count.append(sorted(counted.items()))
            print(sorted(counted.items()))
        # for cross in self.matrix.keys():
        #     print(cross, self.matrix[cross])
        print()
        print()
        print()
        print()
        print()

        self.n_keys = []
        for counted in sort_count:
            self.n_keys.append([])
            for (key,_) in counted:
                self.n_keys[-1].append(key)

        print(self.n_keys)

        print()
        print()
        print()
        print()

        for x1_key in self.n_keys[0]:
            for x2_key in self.n_keys[1]:
                for x3_key in self.n_keys[2]:
                    for x4_key in self.n_keys[3]:
                        key = "".join([x1_key,x2_key,x3_key,x4_key])
                        if not key in self.matrix:
                            self.matrix[key] = 0

km = Count()


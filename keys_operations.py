#!/usr/bin/env python
import random
from help import Help
from reedmuller import reedmuller
#from milner_attack_code import Milner_attack
from milner_attack_code import *

class Generate_Key:
    def __init__(self, rm, secret_key_file, public_key_file):
        self.sk_M = self.gen_random_matrix(rm.k)
        self.sk_P = self.gen_random_permutation(rm.n)
        sk_key_file_desc = open(secret_key_file, 'w')
        for i in range(0, len(self.sk_M)):
            for j in range(0, len(self.sk_M[0])):
                sk_key_file_desc.write(str(self.sk_M[i][j]))
            sk_key_file_desc.write("\n")
        sk_key_file_desc.write("\n")
        for i in range(0, len(self.sk_P)):
            for j in range(0, len(self.sk_P[0])):
                sk_key_file_desc.write(str(self.sk_P[i][j]))
            sk_key_file_desc.write("\n")
        self.G_pub = self.gen_pub_key(rm, self.sk_M, self.sk_P)
        pub_key_file_desc = open(public_key_file, 'w')
        for i in range(0, len(self.G_pub)):
            for j in range(0, len(self.G_pub[0])):
                pub_key_file_desc.write(str(self.G_pub[i][j]))
            pub_key_file_desc.write("\n")
        return
    
    def gen_random_matrix(self, size):
        H = Help()
        flag = False
        while not(flag):
            sk_M = list()
            for j in range(0, size):
                sk_M_row = [random.randint(0, 1) for i in range(0, size)]
                sk_M.append(sk_M_row)
            flag = H.calculate_determenant(sk_M)
        return sk_M

    def gen_random_permutation(self, size):
        list_k = [i for i in range(0, size)]
        list_k_perm = list(list_k)
        random.shuffle(list_k_perm)
        zero_vector = [0 for i in range(0, size)]
        sk_P = [list(zero_vector) for i in range(0, size)]
        for elem in list_k:
            index = list_k_perm.index(elem)
            sk_P[elem][index] = 1
        return sk_P

    def gen_pub_key(self, rm, sk_M, sk_P):
        H = Help()
        matrix = H.mult_matrix_for_matrix(sk_M, rm.matrix_by_row)
        G_pub = H.mult_matrix_for_matrix(matrix, sk_P)
        return G_pub

class Read_Write_keys:
    def __init__(self):
        return

    def read_secret_key(self, in_file):
        in_file_desc = open(in_file, 'r')
        self.sk_M = list()
        for line in in_file_desc:
            if (len(line) > 1):
                sk_M_raw = list()
                for sym in line:
                    if (sym != "\n"):
                        sk_M_raw.append(int(sym))
                self.sk_M.append(sk_M_raw)
            else:
                break
        self.sk_P = list()
        for line in in_file_desc:
            sk_P_raw = list()
            for sym in line:
                if (sym != "\n"):
                    sk_P_raw.append(int(sym))
            self.sk_P.append(sk_P_raw)
        return

    def write_secret_key(self, rec_sk_M, rec_sk_P, out_file):
        sk_key_file_desc = open(out_file, 'w')
        for i in range(0, len(rec_sk_M)):
            for j in range(0, len(rec_sk_M[0])):
                sk_key_file_desc.write(str(rec_sk_M[i][j]))
            sk_key_file_desc.write("\n")
        sk_key_file_desc.write("\n")
        for i in range(0, len(rec_sk_P)):
            for j in range(0, len(rec_sk_P[0])):
                sk_key_file_desc.write(str(rec_sk_P[i][j]))
            sk_key_file_desc.write("\n")
        return

    def read_public_key(self, in_file):
        in_file_desc = open(in_file, 'r')
        self.G_pub = list()
        for line in in_file_desc:
            G_pub_raw = list()
            for sym in line:
                if (sym != "\n"):
                    G_pub_raw.append(int(sym))
            self.G_pub.append(G_pub_raw)
        return

#!/usr/bin/env python
import sys
import random
from help import Help
from combinatorics import Combinatorics
from reedmuller import reedmuller

class Attack_for_r_equal_one:
    def __init__(self, rm, basic_boolean_functions):
        self.rm = rm
        self.G_pub_one = list(basic_boolean_functions)
        return

    def implement_attack(self):
        H = Help()
        a_vec = self.find_a()
        print("a_vec = ", a_vec)
        A_matrix = self.create_A(a_vec)
        permutation = self.find_permutation(A_matrix)
        return permutation

    def find_a(self):
        H = Help()
        size = self.rm.m + 1
        one_vec = [1 for i in range(0, 2**self.rm.m)]
        for dig in range(0, 2**size):
            a_vec = H.convert_decimal_to_binary(dig, size)
            aG_pub = H.mult_vector_for_matrix(a_vec, self.G_pub_one)
            if (aG_pub == one_vec):
                return a_vec
        return None

    def create_A(self, a_vec):
        A_matrix = list()
        size = self.rm.m + 1
        A_matrix.append(list(a_vec))
        for i in range(1, size):
            A_matrix_raw = [0 for j in range(0, size)]
            A_matrix_raw[i] = 1
            A_matrix.append(A_matrix_raw)
        index = a_vec.index(1)
        print("index = ", index)
        A_matrix[index + 1][0] = 1
        A_matrix[index + 1][index + 1] = 0
        return A_matrix

    def find_permutation(self, A_matrix):
        H = Help()
        AG_pub_one = H.mult_matrix_for_matrix(A_matrix, self.G_pub_one)
        rm_1 = reedmuller.ReedMuller(1, self.rm.m)
        permutation = [0 for i in range(0, self.rm.n)]
        print("RM(1, m):")
        for rm_1_raw in rm_1.matrix_by_row:
            print(rm_1_raw)
        print("AG_pub_one:")
        for AG_pub_one_raw in AG_pub_one:
            print(AG_pub_one_raw)
        transpose_AG_pub_one = list(zip(*AG_pub_one))
        j = 0
        for rm_1_transpose_raw in rm_1.M:
            index = transpose_AG_pub_one.index(rm_1_transpose_raw)
            permutation[j] = index
            j += 1
        return permutation

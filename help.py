#!/usr/bin/env python
from combinatorics import Combinatorics
import numpy as np
import galois

"""
ext_rm_R_T = list()
for tup in self.rm.M:
    ext_rm_R_T.append(list(tup))
for raw in ext_rm_R_T:
    raw.append(0)
ext_rm_R_T[7][7] = 1
H.print_matrix(ext_rm_R_T, "ext_rm_R_T")
"""

class Help:
    def neg(self, vec):
        neg_vec = [((vec[i] + 1) % 2) for i in range(0, len(vec))]
        return neg_vec

    def xor(self, vec1, vec2):
        vec = [(vec1[i] + vec2[i]) % 2 for i in range(0, len(vec1))]
        return vec

    def create_affine_sum(self, rm):
        affine_map = dict()
        const_one = [1 for i in range(0, rm.n)]
        RM = list(rm.matrix_by_row)
        for i in range(1, rm.m + 1):
            affine_map[str(RM[i])] = [i - 1]
            neg_RM_i = self.xor(RM[i], const_one)
            affine_map[str(neg_RM_i)] = [i - 1, -1]
        return affine_map

    def extend_tup_matrix_to_n_n(self, tup_matrix, m, n):
        # m < n
        new_matrix = list()
        for tup in tup_matrix:
            new_matrix.append(list(tup))
        self.print_matrix(new_matrix, "new_matrix")
        for j in range(m, n):
            for i in range(0, n):
                new_matrix[i].append(0)
            new_matrix[j][j] = 1
        self.print_matrix(new_matrix, "new_matrix")
        return new_matrix

    def calculate_determenant(self, matrix):
        GF2 = galois.GF(2)
        GF2_matrix = GF2(matrix)
        rank = np.linalg.matrix_rank(GF2_matrix)
        return rank == len(matrix)

    def calculate_factorial(self, n):
        fact = 1
        for i in range(2, n + 1):
            fact = fact * i
        return fact

    def fact(self, n):
        prod = 1
        for i in range(2, n + 1):
            prod = prod * i
        return prod

    def find_binom_border(self, r, m):
        sum_binom = 0
        for j in range(0, r):
            sum_binom += self.binom_calc(m, j)
        iter_size = 2**r - 1
        if sum_binom % iter_size == 0:
            return sum_binom // iter_size
        else:
            return (sum_binom // iter_size) + 1

    def binom_calc(self, n, m):
        return self.fact(n) // (self.fact(n - m) * self.fact(m))

    def calculate_comb(self, n, k):
        comb = 1
        for i in range(n - k + 1, n + 1):
            comb = comb * i
        comb = comb // self.calculate_factorial(k)
        return comb

    def create_inverse_permutation(self, permutation):
        inverse_permutation = [0 for i in range(0, len(permutation))]
        for i in range(0, len(permutation)):
            inverse_permutation[permutation[i]] = int(i)
        return inverse_permutation

    def print_matrix(self, matrix, name):
        print(name + ":")
        for raw in matrix:
            print(raw)
        return

    def sum_of_vectors(self, matrix, combination):
        new_vector = list(matrix[combination[0]])
        for i in range(1, len(combination)):
            for j in range(0, len(new_vector)):
                new_vector[j] = (new_vector[j] + matrix[combination[i]][j]) % 2
        return new_vector

    def convert_decimal_to_binary(self, dec, base):
        string = "{0:{fill" + "}" + str(base) + "b}"
        res_str = string.format(dec, fill='0')
        res_dig = [int(sym) for sym in res_str]
        return res_dig

    def convert_binary_to_decimal(self, binary_list, base):
        dig = 0
        for i in range(0, base):
            dig = dig + binary_list[i] * (2 ** (base - i - 1))
        return dig

    def transpose_matrix(self, matrix):
        n = len(matrix)
        m = len(matrix[0])
        transpose_matrix = list()
        for j in range(0, m):
            curr_list = [ matrix[i][j] for i in range(0, n) ]
            transpose_matrix.append(curr_list)
        return transpose_matrix

    def change_rows_order(self, matrix):
        # m < n
        n = len(matrix)
        m = len(matrix[0])
        for i in range(0, m):
            temp = list(matrix[m - i - 1])
            matrix[m - i - 1] = list(matrix[n - i - 1])
            matrix[n - i - 1] = temp
        return

    def mult_vector_for_matrix(self, vec, matrix):
        res_vec = list()
        sum_ceil = 0
        for j in range(0, len(matrix[0])):
            for i in range(0, len(vec)):
                sum_ceil = (sum_ceil + vec[i] * matrix[i][j]) % 2
            res_vec.append(int(sum_ceil))
            sum_ceil = 0
        return res_vec

    def mult_matrix_for_vector(self, matrix, vec):
        res_vec = list()
        sum_ceil = 0
        for i in range(0, len(matrix)):
            for j in range(0, len(vec)):
                sum_ceil = (sum_ceil + matrix[i][j] * vec[j]) % 2
            res_vec.append(int(sum_ceil))
            sum_ceil = 0
        return res_vec

    def mult_matrix_for_matrix(self, matrix1, matrix2):
        matrix = list()
        n = len(matrix1)
        m = len(matrix1[0])
        k = len(matrix2[0])
        for vec in matrix1:
            res_vec = self.mult_vector_for_matrix(vec, matrix2)
            matrix.append(list(res_vec))
        return matrix

    def find_min_weight_in_depend_list(self, depend_list, size):
        Comb = Combinatorics()
        flag = True
        i = 1
        while (i < size) and (flag):
            allCombinations = []
            Comb.GenerationAllCombinations(allCombinations, size, i)
            #print("allCombinations: ", allCombinations)
            j = 0
            while (j < len(allCombinations)) and (flag):
                curr_list = [0 for x in range(0, size)]
                for index in allCombinations[j]:
                    curr_list[index] = 1
                dec = self.convert_binary_to_decimal(curr_list, size)
                if (depend_list[dec] == 0):
                    flag = False
                    depend_list[dec] = 1
                j = j + 1
            i = i + 1
        return curr_list

    def calculate_weight_for_vector(self, vector):
        weight = vector.count(1)
        return weight
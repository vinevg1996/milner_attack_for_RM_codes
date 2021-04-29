#!/usr/bin/env python
from combinatorics import Combinatorics
import numpy as np
import galois
import random

"""
ext_rm_R_T = list()
for tup in self.rm.M:
    ext_rm_R_T.append(list(tup))
for raw in ext_rm_R_T:
    raw.append(0)
ext_rm_R_T[7][7] = 1
H.print_matrix(ext_rm_R_T, "ext_rm_R_T")
"""

class SLAU:
    def find_first_one_in_col(self, matrix, col):
        for i in range(col, len(matrix)):
            if matrix[i][col] == 1:
                return i
        return -1

    def xor_lines(self, M, row):
        H = Help()
        for i in range(row + 1, len(M)):
            if M[i][row] == 1:
                M[i] = H.xor(M[row], M[i])
        return

    def check_eq(self, A_tup, b, x):
        H = Help()
        A = list()
        for row in A_tup:
            A.append(list(row))
        Ax = H.mult_matrix_for_vector(A, x)
        #print("x = ", x)
        #print("Ax = ", Ax)
        #print("b  = ", b)
        return (Ax == b)

    def gauss(self, A_tup, b):
        H = Help()
        M = list()
        for row in A_tup:
            M.append(list(row))
        #H.print_matrix(M, "M1")
        n = len(M)
        k = len(M[0])
        for i in range(0, len(b)):
            M[i].append(b[i])
        #H.print_matrix(M, "M2")
        for it in range(0, k):
            row = self.find_first_one_in_col(M, it)
            if row != -1:
                #print("row = ", row)
                line = list(M[row])
                M[row] = list(M[it])
                M[it] = list(line)
                self.xor_lines(M, it)
                #H.print_matrix(M, "M_it")
        #H.print_matrix(M, "M")
        square_matrix = list()
        b = list()
        for i in range(0, k):
            square_matrix.append(M[i][0:len(M[i])-1])
            b.append(M[i][len(M[i])-1])
        #H.print_matrix(square_matrix, "square_matrix")
        #print("b = ", b)
        GF2 = galois.GF(2)
        A_galue = GF2(square_matrix)
        b_galue = GF2(b)
        x = np.linalg.solve(A_galue, b_galue)
        list_x = list(np.array(x))
        #print("x = ", list_x)
        return list_x

class Help:
    def choose_independend(self, vectors, rm):
        #print("len = ", len(vectors))
        count = rm.k - self.binom_calc(rm.m, rm.r)
        #print("count = ", count)
        comb = Combinatorics()
        allCombinations = list()
        comb.GenerationAllCombinations(allCombinations, len(vectors), count)
        for comb in allCombinations:
            matrix = list()
            for number in comb:
                matrix.append(list(vectors[number]))
            flag = self.calculate_determenant(matrix)
            #print("flag = ", flag)
            if flag == True:
                #print("flag = ", flag)
                return matrix
        #print("Can't find")
        return None

    def create_g_alpha(self, rm, enc_msg, func_eqs, alpha):
        #self.print_matrix(func_eqs, "func_eqs")
        const_one = [1 for i in range(0, rm.n)]
        const_zero = [0 for i in range(0, rm.n)]
        rem_func = list(const_one)
        for i in range(0, len(func_eqs)):
            func_eq = list(func_eqs[i])
            if func_eq[0] == -1:
                const = 0
                start_id = 1
            else:
                const = 1
                start_id = 0
            if alpha[i] == 1:
                const = (const + 1) % 2
            curr_func = list(const_zero)
            for j in range(start_id, len(func_eq)):
                x = int(func_eq[j])
                curr_func = self.xor(curr_func, rm.matrix_by_row[x + 1])
            if const == 1:
                curr_func = self.neg(curr_func)
            rem_func = self.mult(rem_func, curr_func)
        res_func = self.xor(enc_msg, curr_func)
        return res_func

    def solve_linear_eq(self, rm, func_eqs, alpha):
        size = 2**rm.m
        V_supp = list()
        for dig in range(0, size):
            flag = True
            vars_value = self.convert_decimal_to_binary(dig, rm.m)
            i = 0
            while (i < len(func_eqs)) and (flag):
                if (func_eqs[i][0] == -1):
                    value = (alpha[i] + 1) % 2
                    start_id = 1
                else:
                    value = int(alpha[i])
                    start_id = 0
                sum_eq = 0
                for j in range(start_id, len(func_eqs[i])):
                    x = int(func_eqs[i][j])
                    sum_eq = (sum_eq + vars_value[x]) % 2
                if sum_eq != value:
                    flag = False
                i = i + 1
            if flag == True:
                V_supp.append(list(vars_value))
        return V_supp

    def reverse_vector(self, vec):
        new_vec = [0 for i in range(0, len(vec))]
        for i in range(0, len(vec)):
            new_vec[i] = int(vec[len(vec) - i - 1])
        return new_vec

    def neg(self, vec):
        neg_vec = [((vec[i] + 1) % 2) for i in range(0, len(vec))]
        return neg_vec

    def xor(self, vec1, vec2):
        vec = [(vec1[i] + vec2[i]) % 2 for i in range(0, len(vec1))]
        return vec

    def mult(self, vec1, vec2):
        vec = [(vec1[i] * vec2[i]) % 2 for i in range(0, len(vec1))]
        return vec

    def create_min_word_base(self, rm):
        affine_func_value_list = list()
        affine_vars_list = list()
        const_one = [1 for i in range(0, rm.n)]
        RM = list(rm.matrix_by_row)
        for i in range(1, rm.m + 1):
            affine_func_value_list.append(RM[i])
            affine_vars_list.append([i - 1])
            neg_RM_i = self.neg(RM[i])
            affine_func_value_list.append(neg_RM_i)
            affine_vars_list.append([-1, i - 1])
        for i in range(1, rm.m + 1):
            for j in range(i + 1, rm.m + 1):
                vec = self.xor(RM[i], RM[j])
                affine_func_value_list.append(vec)
                affine_vars_list.append([i - 1, j - 1])
                neg_vec = self.neg(vec)
                affine_func_value_list.append(vec)
                affine_vars_list.append([-1, i - 1, j - 1])
        for i in range(1, rm.m + 1):
            for j in range(i + 1, rm.m + 1):
                for k in range(j + 1, rm.m + 1):
                    vec = self.xor(RM[i], RM[j])
                    vec = self.xor(vec, RM[k])
                    affine_func_value_list.append(vec)
                    affine_vars_list.append([i-1, j-1, k-1])
                    neg_vec = self.neg(vec)
                    affine_func_value_list.append(neg_vec)
                    affine_vars_list.append([-1, i-1, j-1, k-1])
        size = len(affine_func_value_list)
        comb = Combinatorics()
        allCombinations = list()
        comb.GenerationAllCombinations(allCombinations, size, rm.r)
        min_word_base = dict()
        for curr_comb in allCombinations:
            func_value = list(const_one)
            func_vars = list()
            for number in curr_comb:
                func_value = self.mult(func_value, affine_func_value_list[number])
                func_vars.append(affine_vars_list[number])
            min_word_base[str(func_value)] = func_vars
        return min_word_base

    def extend_gen_matrix_to_n_n(self, rm):
        RM = list(rm.matrix_by_row)
        GF2 = galois.GF(2)
        for i in range(rm.k, rm.n):
            print("i = ", i)
            line = [random.randint(0, 1) for i in range(0, rm.n)]
            full_rank_flag = False
            while not(full_rank_flag):
                matrix = list(RM)
                matrix.append(line)
                GF2_matrix = GF2(matrix)
                rank = np.linalg.matrix_rank(GF2_matrix)
                if (rank == len(matrix)):
                    RM.append(line)
                    full_rank_flag = True
        self.print_matrix(RM, "RM")
        tup_matrix = list(zip(*RM))
        new_matrix = list()
        for tup in tup_matrix:
            new_matrix.append(list(tup))
        return new_matrix

    def extend_tup_matrix_to_n_n(self, rm):
        # m < n
        new_matrix = list()
        for tup in tup_matrix:
            new_matrix.append(list(tup))
        GF2 = galois.GF(2)
        GF2_matrix = GF2(new_matrix)
        rank = np.linalg.matrix_rank(GF2_matrix)
        print("rank = ", rank)
        print("len_0 = ", len(new_matrix[0]))
        self.print_matrix(new_matrix, "new_matrix")
        for j in range(m, n):
            for i in range(0, n):
                new_matrix[i].append(0)
        #self.print_matrix(new_matrix, "new_matrix1")
        for i in range(m, n):
            #for k in range(i, n):
            new_matrix[i][i] = 1
        self.print_matrix(new_matrix, "new_matrix2")
        
        GF2_matrix2 = GF2(new_matrix)
        rank2 = np.linalg.matrix_rank(GF2_matrix2)
        print("rank2 = ", rank2)
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
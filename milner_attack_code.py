#!/usr/bin/env python
import sys
import random
from help import *
from combinatorics import Combinatorics
from reedmuller import reedmuller
from attack_for_r_equal_one import Attack_for_r_equal_one
import numpy as np
import galois

class Milner_attack:
    def __init__(self, rm, G_pub):
        self.H = Help()
        self.rm = rm
        self.G_pub = G_pub
        return

    def bool_funcs_from_min_weight_word(self, enc_msg, func_vars):

        return

    def find_minimal_weight_monoms(self, count_base_vectors):
        min_weight_words = list()
        enough_vectors = False
        start_id = 1
        while not(enough_vectors):
            [dig, answer] = self.find_minimal_weight_word(self.rm, self.G_pub, start_id)
            if answer == "yes":
                msg = H.convert_decimal_to_binary(dig, self.rm.k)
                enc_msg = H.mult_vector_for_matrix(msg, self.G_pub)
                func_vars = self.min_word_base[str(enc_msg)]
                #min_weight_words.append([enc_msg, func_vars])
                min_weight_word = list(min_weight_words[i][0])
                func_vars = list(min_weight_words[i][1])
                boolean_functions = self.create_vectors_for_basis(min_weight_word, func_vars)
                bool_funcs = self.bool_funcs_from_min_weight_word(enc_msg, func_vars)
                
                start_id = dig + 1
        return

    def create_basis_for_r_minus_1(self):
        self.min_word_base = self.H.create_min_word_base(self.rm)
        min_weight_words = self.find_monom_minimal_weight_words(count)

        return

    def implement_attack(self):
        self.min_word_base = self.H.create_min_word_base(self.rm)
        #count = self.H.find_binom_border(self.rm.r, self.rm.m)
        count = 4
        print("count = ", count)
        min_weight_words = self.find_monom_minimal_weight_words(count)
        print("min_weight_words:")
        print(min_weight_words[0])
        print(min_weight_words[1])
        print("len_min_weight_words = ", len(min_weight_words))
        if (len(min_weight_words) < count):
            print("Can't find enough min_weight_words")
            return [None, None]
        print("Start attack:")
        basic_boolean_functions = self.create_r_minus_1_basic(min_weight_words, self.rm.r)
        if (self.rm.r == 2):
            A_RM_1 = Attack_for_r_equal_one(self.rm, basic_boolean_functions)
            permutation = A_RM_1.implement_attack()
            print("permutation = ", permutation)
            zero_vector = [0 for i in range(0, len(permutation))]
            M_P = [list(zero_vector) for i in range(0, len(permutation))]
            for elem in permutation:
                index = permutation.index(elem)
                M_P[elem][index] = 1
            inverse_permutation = self.H.create_inverse_permutation(permutation)
            M_P_inv = [list(zero_vector) for i in range(0, len(inverse_permutation))]
            for elem in inverse_permutation:
                index = inverse_permutation.index(elem)
                M_P_inv[elem][index] = 1
            self.H.print_matrix(M_P_inv, "M_P_inv")
            recover_sk_M = self.find_M_with_linalg(M_P_inv)
            return [recover_sk_M, M_P]
        return

    def find_monom_minimal_weight_words(self, count):
        H = Help()
        start_id = 1
        min_weight_words = list()
        for curr_id in range(0, count):
            [dig, answer] = self.find_minimal_weight_word(self.rm, self.G_pub, start_id)
            if answer == "yes":
                msg = H.convert_decimal_to_binary(dig, self.rm.k)
                enc_msg = H.mult_vector_for_matrix(msg, self.G_pub)
                func_vars = self.min_word_base[str(enc_msg)]
                min_weight_words.append([enc_msg, func_vars])
                start_id = dig + 1
        return min_weight_words

    def find_enc_msg_in_min_word_base(self, enc_msg):
        if (str(enc_msg) in self.min_word_base.keys()):
            return True
        return False

    def find_minimal_weight_word(self, rm, G_pub, start_id):
        H = Help()
        min_weight = 2 ** (rm.m - rm.r)
        for dig in range(start_id, 2**rm.k):
            msg = H.convert_decimal_to_binary(dig, rm.k)
            enc_msg = H.mult_vector_for_matrix(msg, G_pub)
            if H.calculate_weight_for_vector(enc_msg) == min_weight:
                flag = self.find_enc_msg_in_min_word_base(enc_msg)
                if flag == True:
                    return [dig, "yes"]
        return [dig, "no"]

    def create_r_minus_1_basic(self, min_weight_words, r):
        H = Help()
        count_basic_vectors = 0
        for i in range(0, r):
            count_basic_vectors += H.binom_calc(self.rm.m, i)
        basic_boolean_functions = list()
        curr_basic_vectors = 0
        for i in range(0, len(min_weight_words)):
            min_weight_word = list(min_weight_words[i][0])
            func_vars = list(min_weight_words[i][1])
            boolean_functions = self.create_vectors_for_basis(min_weight_word, func_vars)
            self.H.print_matrix(boolean_functions, "boolean_functions")
            for bool_func in boolean_functions:
                basic_boolean_functions.append(bool_func)
        print("independend_bool_func:")
        independend_bool_func = self.H.choose_independend(basic_boolean_functions, self.rm)
        return independend_bool_func

    def create_vectors_for_basis(self, min_weight_word, func_vars):
        boolean_functions = list()
        alpha_one = [1 for i in range(0, self.rm.r)]
        V_supp_one = self.H.solve_linear_eq(self.rm, func_vars, alpha_one)
        print("V_supp_one = ", V_supp_one)
        pat_boolean_function = [0 for i in range(0, self.rm.n)]
        for bool_set in V_supp_one:
            rev_bool_set = self.H.reverse_vector(bool_set)
            index = self.H.convert_binary_to_decimal(rev_bool_set, len(rev_bool_set))
            pat_boolean_function[index] = 1
        size = 2**self.rm.r - 1
        for dig in range(0, size):
            alpha = self.H.convert_decimal_to_binary(dig, self.rm.r)
            V_supp = self.H.solve_linear_eq(self.rm, func_vars, alpha)
            print("V_supp = ", V_supp)
            boolean_function = list(pat_boolean_function)
            for bool_set in V_supp:
                rev_bool_set = self.H.reverse_vector(bool_set)
                index = self.H.convert_binary_to_decimal(rev_bool_set, len(rev_bool_set))
                boolean_function[index] = 1
            boolean_functions.append(boolean_function)
        return boolean_functions

    def create_basic_vectors(self, min_weight_word, func_vars, r):
        H = Help()
        comb = Combinatorics()
        allCombinations = list()
        comb.GenerationAllCombinations(allCombinations, self.rm.m, r)
        sum_comb_r_minus_1 = 0
        for i in range(0, r):
            sum_comb_r_minus_1 = sum_comb_r_minus_1 + H.binom_calc(self.rm.m, i)
        new_index = index - sum_comb_r_minus_1
        variables = list(allCombinations[new_index])
        supp_sets_variables = self.create_V_sets(variables)
        boolean_functions = list()
        for supp_sets in supp_sets_variables:
            boolean_function = [0 for i in range(0, self.rm.n)]
            for supp_set in supp_sets:
                dig = H.convert_binary_to_decimal(supp_set, len(supp_set))
                boolean_function[dig] = 1
            boolean_functions.append(boolean_function)
        return boolean_functions

    def create_V_sets(self, variables):
        H = Help()
        supp_sets = list()
        alpha_one = [1 for i in range(0, len(variables))]
        supp_sets_one = self.create_V_set(variables, alpha_one)
        for dig in range(0, 2**len(variables) - 1):
            alpha = H.convert_decimal_to_binary(dig, len(variables))
            alpha_supp_sets = self.create_V_set(variables, alpha)
            for alpha_supp_set in alpha_supp_sets:
                curr_supp_set = list(supp_sets_one)
                for curr_set in alpha_supp_sets:
                    curr_supp_set.append(list(curr_set))
            supp_sets.append(curr_supp_set)
        return supp_sets

    def create_V_set(self, variables, alpha):
        H = Help()
        supp_sets = list()
        compl_variables = list()
        for i in range(0, self.rm.m):
            if not(i in variables):
                compl_variables.append(i)
        supp_set_pattern = [0 for i in range(0, self.rm.m)]
        for i in range(0, len(variables)):
            supp_set_pattern[variables[i]] = int(alpha[i])
        for dig in range(0, 2**len(compl_variables)):
            compl_variables_values = H.convert_decimal_to_binary(dig, len(compl_variables))
            supp_set = list(supp_set_pattern)
            for i in range(0, len(compl_variables)):
                supp_set[compl_variables[i]] = int(compl_variables_values[i])
            supp_sets.append(list(supp_set))
        return supp_sets

    def find_M_with_linalg(self, M_P_inv):
        H = Help()
        B = H.mult_matrix_for_matrix(self.G_pub, M_P_inv)
        #H.print_matrix(B, "B")
        #H.print_matrix(self.rm.matrix_by_row, "matrix_by_row")
        slau = SLAU()
        sk_M = list()
        for i in range(0, self.rm.k):
            x = slau.gauss(self.rm.M, B[i])
            sk_M.append(x)
            print("i = ", i, ": x = ", x)
        H.print_matrix(sk_M, "sk_M")
        return sk_M

    def find_M(self, M_P_inv):
        H = Help()
        B = H.mult_matrix_for_matrix(self.G_pub, M_P_inv)
        print("B:")
        for B_raw in B:
            print(B_raw)
        new_M = list()
        for i in range(0, self.rm.k):
            flag = False
            for dig in range(0, 2**self.rm.k):
                M_raw = H.convert_decimal_to_binary(dig, self.rm.k)
                M_raw_mult_R = H.mult_vector_for_matrix(M_raw, self.rm.matrix_by_row)
                if (M_raw_mult_R == B[i]):
                    new_M.append(M_raw)
                    flag = True
                    break
        print("new_M:")
        for M_raw in new_M:
            print(M_raw)
        return new_M

    def check_function(self, M, permutation):
        H = Help()
        zero_vector = [0 for i in range(0, len(permutation))]
        M_P = [list(zero_vector) for i in range(0, len(permutation))]
        for elem in permutation:
            index = permutation.index(elem)
            M_P[elem][index] = 1
        matrix_1 = H.mult_matrix_for_matrix(M, self.rm.matrix_by_row)
        check_G_pub = H.mult_matrix_for_matrix(matrix_1, M_P)
        print(check_G_pub == self.G_pub)
        return

"""
x0 = self.rm.matrix_by_row[1]
x1 = self.rm.matrix_by_row[2]
x2 = self.rm.matrix_by_row[3]
x3 = self.rm.matrix_by_row[4]
print("x1 =", x1)
print("x0 =", x0)
print("x2 =", x2)
print("x3 =", x3)
neg_x1 = self.H.neg(x1)
vec = self.H.xor(x0, x2)
vec = self.H.xor(vec, x3)
print("vec =", vec)
print("____vec =", vec)
print("neg__x1 =", neg_x1)
const_one = [1 for i in range(0, rm.n)]
mult_vec = list(const_one)
mult_vec = self.H.mult(mult_vec, neg_x1)
mult_vec = self.H.mult(mult_vec, vec)
print("multVec =", mult_vec)
"""
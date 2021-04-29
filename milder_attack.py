#!/usr/bin/env python
import sys
import random
from help import *
from combinatorics import Combinatorics
from reedmuller import reedmuller
from attack_for_r_equal_one import Attack_for_r_equal_one
import numpy as np
import galois

class Attack_of_Milder:
    def __init__(self, rm, G_pub):
        self.H = Help()
        self.rm = rm
        self.G_pub = G_pub
        return

    def create_r_minus_1_basic(self, count):
        min_weight_words = list()
        boolean_functions = list()
        #is_enough_vectors = False
        start_id = 1
        while True:
            [dig, answer] = self.find_minimal_weight_word(self.rm, self.G_pub, start_id)
            if answer == "yes":
                msg = self.H.convert_decimal_to_binary(dig, self.rm.k)
                enc_msg = self.H.mult_vector_for_matrix(msg, self.G_pub)
                func_vars = self.min_word_base[str(enc_msg)]
                bool_funcs = self.bool_funcs_from_min_weight_word(enc_msg, func_vars)
                #bool_funcs = self.g_funcs_from_min_weight_word(enc_msg, func_vars)
                for bool_func in bool_funcs:
                    boolean_functions.append(bool_func)
                if (len(boolean_functions) >= count):
                    basic_boolean_functions = self.H.choose_independend(boolean_functions, self.rm)
                    if (basic_boolean_functions != None):
                        #self.H.print_matrix(basic_boolean_functions, "basic_boolean_functions")
                        return basic_boolean_functions
                start_id = dig + 1
        return None

    def implement_attack(self):
        self.min_word_base = self.H.create_min_word_base(self.rm)
        count = self.rm.k - self.H.binom_calc(self.rm.m, self.rm.r)
        basic_boolean_functions = self.create_r_minus_1_basic(count)
        A_RM_1 = Attack_for_r_equal_one(self.rm, basic_boolean_functions)
        permutation = A_RM_1.implement_attack()
        #print("permutation = ", permutation)
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
        #self.H.print_matrix(M_P_inv, "M_P_inv")
        recover_sk_M = self.find_M_with_linalg(M_P_inv)
        #recover_sk_M = self.find_M(M_P_inv)
        return [recover_sk_M, M_P, M_P_inv]

    def g_funcs_from_min_weight_word(self, min_weight_word, func_vars):
        boolean_functions = list()
        size = 2**self.rm.r - 1
        for dig in range(0, size):
            alpha = self.H.convert_decimal_to_binary(dig, self.rm.r)
            res_func = self.H.create_g_alpha(self.rm, min_weight_word, func_vars, alpha)
            boolean_functions.append(res_func)
        return boolean_functions

    def bool_funcs_from_min_weight_word(self, min_weight_word, func_vars):
        boolean_functions = list()
        alpha_one = [1 for i in range(0, self.rm.r)]
        V_supp_one = self.H.solve_linear_eq(self.rm, func_vars, alpha_one)
        #print("V_supp_one = ", V_supp_one)
        pat_boolean_function = [0 for i in range(0, self.rm.n)]
        for bool_set in V_supp_one:
            rev_bool_set = self.H.reverse_vector(bool_set)
            index = self.H.convert_binary_to_decimal(rev_bool_set, len(rev_bool_set))
            pat_boolean_function[index] = 1
        size = 2**self.rm.r - 1
        for dig in range(0, size):
            alpha = self.H.convert_decimal_to_binary(dig, self.rm.r)
            V_supp = self.H.solve_linear_eq(self.rm, func_vars, alpha)
            #print("V_supp = ", V_supp)
            boolean_function = list(pat_boolean_function)
            for bool_set in V_supp:
                rev_bool_set = self.H.reverse_vector(bool_set)
                index = self.H.convert_binary_to_decimal(rev_bool_set, len(rev_bool_set))
                boolean_function[index] = 1
            boolean_functions.append(boolean_function)
        return boolean_functions

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

    def find_M_with_linalg(self, M_P_inv):
        H = Help()
        B = H.mult_matrix_for_matrix(self.G_pub, M_P_inv)
        #H.print_matrix(B, "B")
        #H.print_matrix(self.rm.matrix_by_row, "matrix_by_row")
        slau = SLAU()
        sk_M = list()
        
        #print("+++++++++++++++++++++")
        #for i in range(0, 1):
        for i in range(0, self.rm.k):
            x = slau.gauss(self.rm.M, B[i])
            check_flag = slau.check_eq(self.rm.M, B[i], x)
            #print("i = ", i)
            #print("check_flag = ", check_flag)
            sk_M.append(x)
            #print("i = ", i, ": x = ", x)
        H.print_matrix(sk_M, "sk_M")
        return sk_M

    def find_M(self, M_P_inv):
        H = Help()
        B = H.mult_matrix_for_matrix(self.G_pub, M_P_inv)
        #self.H.print_matrix(B, "B")
        new_M = list()
        for i in range(0, self.rm.k):
        #for i in range(0, 1):
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

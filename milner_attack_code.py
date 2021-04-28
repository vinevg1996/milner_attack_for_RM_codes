#!/usr/bin/env python
import sys
import random
from help import Help
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

    def implement_attack(self):
        affine_map = self.H.create_affine_sum(self.rm)
        for value in affine_map.keys():
            print(value, ":", affine_map[value])
        count = self.H.find_binom_border(self.rm.r, self.rm.m)
        print("count = ", count)
        min_weight_words = self.find_monom_minimal_weight_words(count)
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
                index = self.rm.matrix_by_row.index(enc_msg)
                min_weight_words.append([enc_msg, index])
                start_id = dig + 1
        return min_weight_words

    def find_enc_msg_in_base_code(self, enc_msg):
        if (enc_msg in self.rm.matrix_by_row):
            index = self.rm.matrix_by_row.index(enc_msg)
            return True
        return False

    def find_minimal_weight_word(self, rm, G_pub, start_id):
        H = Help()
        min_weight = 2 ** (rm.m - rm.r)
        for dig in range(start_id, 2**rm.k):
            msg = H.convert_decimal_to_binary(dig, rm.k)
            enc_msg = H.mult_vector_for_matrix(msg, G_pub)
            if H.calculate_weight_for_vector(enc_msg) == min_weight:
                #print("enc_msg = ", enc_msg)
                flag = self.find_enc_msg_in_base_code(enc_msg)
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
            index = int(min_weight_words[i][1])
            boolean_functions = self.create_basic_vectors(min_weight_word, index, r)
            if (curr_basic_vectors + len(boolean_functions) < count_basic_vectors):
                for boolean_function in boolean_functions:
                    basic_boolean_functions.append(list(boolean_function))
                curr_basic_vectors += len(boolean_functions)
            else:
                j = 0
                while curr_basic_vectors < count_basic_vectors:
                    basic_boolean_functions.append(list(boolean_functions[j]))
                    j = j + 1
                    curr_basic_vectors += 1
        return basic_boolean_functions

    def create_basic_vectors(self, min_weight_word, index, r):
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
        GF2 = galois.GF(2)
        ext_rm_R_T = H.extend_tup_matrix_to_n_n(self.rm.M, self.rm.k, self.rm.n)
        ext_rm_M = GF2(ext_rm_R_T)
        sk_M = list()
        for i in range(0, self.rm.k):
            b = GF2(B[i])
            x = np.linalg.solve(ext_rm_M, b)
            x_list = list(np.array(x))
            sk_M.append(x_list[0:self.rm.k])
            #print("i = ", i, np.array(x))
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

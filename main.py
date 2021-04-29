#!/usr/bin/env python
import sys
import random
from help import *
from reedmuller import reedmuller
#from milner_attack_code import *
from milder_attack import *
from keys_operations import *

#python3 main.py mode=1 2 4 secret_key.txt public_key.txt
#python3 main.py mode=2 public_key.txt 2 4 recover_secret_key.txt
#python3 main.py mode=3 recover_secret_key.txt public_key.txt 2 4

H = Help()

def is_attack_corret(rm, M, P, G_pub):
    matrix_1 = H.mult_matrix_for_matrix(M, rm.matrix_by_row)
    check_G_pub = H.mult_matrix_for_matrix(matrix_1, P)
    H.print_matrix(matrix_1, "matrix_1")
    H.print_matrix(check_G_pub, "check_G_pub")
    return (check_G_pub == G_pub)

def check_attack(rm, recover_sk_M, M_P, G_pub, M_P_inv):
    print("*********************")
    matrix_1 = H.mult_matrix_for_matrix(recover_sk_M, rm.matrix_by_row)
    H.print_matrix(matrix_1, "matrix_1")
    matrix_2 = H.mult_matrix_for_matrix(G_pub, M_P_inv)
    H.print_matrix(matrix_2, "matrix_2")
    return

if (sys.argv[1] == "mode=1"):
    r = int(sys.argv[2])
    m = int(sys.argv[3])
    secret_key_file = str(sys.argv[4])
    public_key_file = str(sys.argv[5])
    rm = reedmuller.ReedMuller(r, m)
    gen_key = Generate_Key(rm, secret_key_file, public_key_file)
elif (sys.argv[1] == "mode=2"):
    public_key_file = str(sys.argv[2])
    r = int(sys.argv[3])
    m = int(sys.argv[4])
    out_secret_key_file = str(sys.argv[5])
    rm = reedmuller.ReedMuller(r, m)
    rk = Read_Write_keys()
    rk.read_public_key(public_key_file)
    #M = Milner_attack(rm, rk.G_pub)
    M = Attack_of_Milder(rm, rk.G_pub)
    [recover_sk_M, M_P, M_P_inv] = M.implement_attack()
    #check_attack(rm, recover_sk_M, M_P, rk.G_pub, M_P_inv)
    rk.write_secret_key(recover_sk_M, M_P, out_secret_key_file)
elif (sys.argv[1] == "mode=3"):
    recover_secret_key_file = str(sys.argv[2])
    public_key_file = str(sys.argv[3])
    rk = Read_Write_keys()
    rk.read_secret_key(recover_secret_key_file)
    rk.read_public_key(public_key_file)
    r = int(sys.argv[4])
    m = int(sys.argv[5])
    rm = reedmuller.ReedMuller(r, m)
    flag = is_attack_corret(rm, rk.sk_M, rk.sk_P, rk.G_pub)
    print("does correct attack?", flag)
else:
    print("start_gauss")
    RM = [[1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 0, 0, 1, 1, 0, 0],
          [1, 0, 1, 0, 1, 0, 1, 0],
          [1, 1, 0, 0, 0, 0, 0, 0],
          [1, 0, 1, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 1, 0, 0, 0]]
    B = [[0, 1, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [1, 0, 0, 1, 1, 1, 1, 1],
         [1, 0, 0, 1, 0, 1, 1, 0],
         [0, 1, 1, 0, 0, 1, 1, 0],
         [1, 1, 0, 1, 0, 0, 0, 1]]
    RM_T = list(zip(*RM))
    S = SLAU()
    S.gauss(RM_T, B[0])
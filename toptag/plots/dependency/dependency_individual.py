import numpy as np
import matplotlib.pyplot as plt
import re
import os

#pfn teacher auc save path
pfn_teacher_auc_path = '/users/yzhou276/work/toptag/simple/pfn/auc/'

#dnn simple auc save path
dnn_simple_auc_path = '/users/yzhou276/work/toptag/simple/dnn/auc/'

#dnn student auc save path
dnn_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/auc/'

#dnn student avg multi distill auc save path
dnn_avg_multi_auc_path = '/users/yzhou276/work/toptag/student/dnn/avg_auc/'

#dnn student lms multi distill auc save path
dnn_lms_multi_auc_path = '/users/yzhou276/work/toptag/student/dnn/lms_auc/'

#to save path
to_save_path = '/users/yzhou276/work/toptag/plots/dependency/dependency_plot/'

number_pattern = r"[-+]?\d*\.\d+|\d+"

#Load pfn teacher
latent_size = [128]#[16,32,64,128,256]
mix_latent_size = [128]
phi_size = [250]

pythia_pfn_auc = []
pythia_pfn_dependency = []
herwig_pfn_auc = []
herwig_pfn_dependency = []
mix_pfn_auc = []
mix_pfn_dependency = []
for a in phi_size:
    for b in latent_size:
        with open(pfn_teacher_auc_path+f'best_pythia_pfn_latent{b}_phi{a}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_pfn_dependency.append(float(dependency[0]))
        with open(pfn_teacher_auc_path+f'best_herwig_pfn_latent{b}_phi{a}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_pfn_dependency.append(float(dependency[0]))
    for b in mix_latent_size:
        with open(pfn_teacher_auc_path+f'best_mix_pfn_latent{b}_phi{a}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            mix_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            mix_pfn_dependency.append(float(dependency[0]))

#Load dnn simple
nlayers = 2 #2, 5, 3
dense_sizes = 100 #100, 25, 10

pythia_dnn_auc = []
pythia_dnn_dependency = []
herwig_dnn_auc = []
herwig_dnn_dependency = []
mix_dnn_auc = []
mix_dnn_dependency = []

with open(dnn_simple_auc_path+f'best_pythia_dnn_nlayers{nlayers}_dense{dense_sizes}.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    pythia_dnn_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    pythia_dnn_dependency.append(float(dependency[0]))

with open(dnn_simple_auc_path+f'best_herwig_dnn_nlayers{nlayers}_dense{dense_sizes}.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    herwig_dnn_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    herwig_dnn_dependency.append(float(dependency[0]))

with open(dnn_simple_auc_path+f'best_mix_dnn_nlayers{nlayers}_dense{dense_sizes}.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    mix_dnn_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    mix_dnn_dependency.append(float(dependency[0]))


#Load dnn single distill student
pythia_dnn_single_distill_auc = []
pythia_dnn_single_distill_dependency = []
herwig_dnn_single_distill_auc = []
herwig_dnn_single_distill_dependency = []
mix_dnn_single_distill_auc = []
mix_dnn_single_distill_dependency = []

with open(dnn_student_auc_path+f'best_pythia_dnn_student_nlayers{nlayers}_dense{dense_sizes}.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    pythia_dnn_single_distill_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    pythia_dnn_single_distill_dependency.append(float(dependency[0]))

with open(dnn_student_auc_path+f'best_herwig_dnn_student_nlayers{nlayers}_dense{dense_sizes}.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    herwig_dnn_single_distill_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    herwig_dnn_single_distill_dependency.append(float(dependency[0]))

with open(dnn_student_auc_path+f'best_mix_dnn_student_nlayers{nlayers}_dense{dense_sizes}.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    mix_dnn_single_distill_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    mix_dnn_single_distill_dependency.append(float(dependency[0]))

'''
#Load dnn avg multi distill student
pythia_num = 1
herwig_num = 1
mix_num = 1

mix_dnn_avg_multi_distill_auc = []
mix_dnn_avg_multi_distill_dependency = []
mix_dnn_lms_multi_distill_auc = []
mix_dnn_lms_multi_distill_dependency = []

with open(dnn_avg_multi_auc_path+f'best_mix_dnn_student_nlayers{nlayers}_dense{dense_sizes}_by_{pythia_num}pythia_{herwig_num}herwig_{mix_num}mix_pfn_teachers.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    mix_dnn_avg_multi_distill_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    mix_dnn_avg_multi_distill_dependency.append(float(dependency[0]))

with open(dnn_lms_multi_auc_path+f'best_mix_dnn_student_nlayers{nlayers}_dense{dense_sizes}_by_{pythia_num}pythia_{herwig_num}herwig_{mix_num}mix_pfn_teachers.txt', 'r') as file:
    lines = file.readlines()
    pythia_auc = re.findall(number_pattern, lines[0])
    mix_dnn_lms_multi_distill_auc.append(float(pythia_auc[1]))
    dependency = re.findall(number_pattern, lines[2])
    mix_dnn_lms_multi_distill_dependency.append(float(dependency[0]))
'''

#Pareto Plot
plt.figure()
plt.scatter(pythia_pfn_auc, pythia_pfn_dependency, label = 'Pythia Teacher PFN', marker = '*', color = 'red', s = 100)
plt.scatter(herwig_pfn_auc, herwig_pfn_dependency, label = 'Herwig Teacher PFN', marker = '*', color = 'blue', s = 100)
plt.scatter(mix_pfn_auc, mix_pfn_dependency, label = 'Mix Teacher PFN', marker = '*', color = 'orange', s = 100)
plt.scatter(pythia_dnn_auc, pythia_dnn_dependency, label = 'Pythia Simple DNN', marker = 'x', color = 'red', s = 100)
plt.scatter(herwig_dnn_auc, herwig_dnn_dependency, label = 'Herwig Simple DNN', marker = 'x', color = 'blue', s = 100)
plt.scatter(mix_dnn_auc, mix_dnn_dependency, label = 'Mix Simple DNN', marker = 'x', color = 'orange', s = 100)
plt.scatter(pythia_dnn_single_distill_auc, pythia_dnn_single_distill_dependency, label = 'Pythia Student DNN', marker = '.', color = 'red', s = 100)
plt.scatter(herwig_dnn_single_distill_auc, herwig_dnn_single_distill_dependency, label = 'Herwig Student DNN', marker = '.', color = 'blue', s = 100)
plt.scatter(mix_dnn_single_distill_auc, mix_dnn_single_distill_dependency, label = 'Mix Student DNN', marker = '.', color = 'orange', s = 100)
#plt.scatter(mix_dnn_avg_multi_distill_auc, mix_dnn_avg_multi_distill_dependency, label = f'Student by {pythia_num+herwig_num+mix_num} Teachers (avg)', marker = '.', color = 'magenta', s = 100)
#plt.scatter(mix_dnn_lms_multi_distill_auc, mix_dnn_lms_multi_distill_dependency, label = f'Student by {pythia_num+herwig_num+mix_num} Teachers (lms)', marker = '.', color = 'pink', s = 100)
plt.title(f'{(dense_sizes,)*nlayers}')
plt.xlabel('AUC on Pythia dataset')
plt.ylabel('AUC_Diff / AUC_on_Pythia')
plt.legend(loc=(0.01,0.1))
plt.show()
plt.savefig(to_save_path + f'{(dense_sizes,)*nlayers}_mix_plot_multi.jpg')

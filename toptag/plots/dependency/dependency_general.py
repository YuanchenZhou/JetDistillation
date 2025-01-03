import numpy as np
import matplotlib.pyplot as plt
import re
import os

#pfn teacher auc save path
pfn_teacher_auc_path = '/users/yzhou276/work/toptag/simple/pfn/auc/'

#dnn simple auc save path
dnn_simple_auc_path = '/users/yzhou276/work/toptag/simple/dnn/auc/'

#efn simple auc save path
efn_simple_auc_path = '/users/yzhou276/work/toptag/simple/efn/auc/'

#dnn student auc save path
dnn_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/auc/'

#dnn avg multi student auc save path
dnn_avg_multi_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/avg_auc/'

#dnn lms multi student auc save path
dnn_lms_multi_student_auc_path = '/users/yzhou276/work/toptag/student/dnn/lms_auc/'

#efn student auc save path
efn_student_auc_path = '/users/yzhou276/work/toptag/student/efn/auc/'

#to save path
to_save_path = '/users/yzhou276/work/toptag/plots/dependency/dependency_plot/'

number_pattern = r"[-+]?\d*\.\d+|\d+"

#Load pfn teacher auc
pfn_latent_sizes = [1,2,4,8,16,32,64,128,256,512]
pfn_phi_sizes = [50,100,250,500]
pythia_pfn_auc = [] # Pythia pfn AUC on pythia dataset
pythia_pfn_dependency = [] # AUC_diff / pythia_auc
herwig_pfn_auc = [] # Herwig pfn AUC on pythia dataset
herwig_pfn_dependency = [] # AUC_diff / pythia_auc
for a in pfn_latent_sizes:
    for b in pfn_phi_sizes:
        with open(pfn_teacher_auc_path+f'best_pythia_pfn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_pfn_dependency.append(float(dependency[0]))
        with open(pfn_teacher_auc_path+f'best_herwig_pfn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_pfn_dependency.append(float(dependency[0]))

'''
pfn_mix_latent_sizes = [128]
pfn_mix_phi_sizes = [250]
mix_pfn_auc = []
mix_pfn_dependency = []
for a in pfn_mix_latent_sizes:
    for b in pfn_mix_phi_sizes:
        with open(pfn_teacher_auc_path+f'best_mix_pfn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            mix_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            mix_pfn_dependency.append(float(dependency[0]))
'''

#Load dnn simple auc
dnn_simple_nlayers = [2,3,4,5,10]
dnn_simple_dense_sizes = [1,10,25,50,100,200,300,500]
pythia_dnn_auc = []
pythia_dnn_dependency = []
herwig_dnn_auc = []
herwig_dnn_dependency = []
for a in dnn_simple_nlayers:
    for b in dnn_simple_dense_sizes:
        with open(dnn_simple_auc_path+f'best_pythia_dnn_nlayers{a}_dense{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_dnn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_dnn_dependency.append(float(dependency[0]))
        with open(dnn_simple_auc_path+f'best_herwig_dnn_nlayers{a}_dense{b}.txt', 'r')	as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_dnn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_dnn_dependency.append(float(dependency[0]))

'''
dnn_mix_simple = ['nlayers2_dense100', 'nlayers3_dense10', 'nlayers5_dense25']
'''

#Load efn simple auc
efn_latent_sizes = [1,2,4,8,16,32,64,128,256,512]
efn_phi_sizes = [50,100,250,500]
pythia_efn_auc = []
pythia_efn_dependency = []
herwig_efn_auc = []
herwig_efn_dependency = []
for a in efn_latent_sizes:
    for b in efn_phi_sizes:
        with open(efn_simple_auc_path+f'best_pythia_efn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_efn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_efn_dependency.append(float(dependency[0]))
        with open(efn_simple_auc_path+f'best_herwig_efn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_efn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_efn_dependency.append(float(dependency[0]))

#Pareto Plot
plt.figure()
plt.scatter(pythia_pfn_auc, pythia_pfn_dependency, label = 'Pythia PFN', marker = '.', color = 'blue')
plt.scatter(herwig_pfn_auc, herwig_pfn_dependency, label = 'Herwig PFN', marker = '.')
plt.scatter(pythia_dnn_auc, pythia_dnn_dependency, label = 'Pythia DNN', marker = '.')
plt.scatter(herwig_dnn_auc, herwig_dnn_dependency, label = 'Herwig DNN', marker = '.')
plt.scatter(pythia_efn_auc, pythia_efn_dependency, label = 'Pythia EFN', marker = '.')
plt.scatter(herwig_efn_auc, herwig_efn_dependency, label = 'Herwig EFN', marker = '.')
plt.title('General Dependency Plot')
plt.xlabel('AUC on Pythia dataset')
plt.ylabel('AUC_Diff / AUC_on_Pythia')
plt.xlim(0.85,0.96)
plt.ylim(0,0.03)
plt.legend()
plt.show()
plt.savefig(to_save_path + 'general_dependency_plot.jpg')

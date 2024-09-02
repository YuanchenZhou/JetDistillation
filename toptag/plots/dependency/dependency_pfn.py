import numpy as np
import matplotlib.pyplot as plt
import re
import os

#pfn teacher auc save path
pfn_teacher_auc_path = '/users/yzhou276/work/toptag/simple/pfn/auc/'

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
mix_pfn_auc = []
mix_pfn_dependency = []

colors = ['blue','green','orange','purple','red']
alpha_values = np.linspace(0,1,len(pfn_latent_sizes)+1)

fig,ax = plt.subplots(figsize=(16,7))
for n in range(len(pfn_phi_sizes)):
    b = pfn_phi_sizes[n]
    for m in range(len(pfn_latent_sizes)):
        a = pfn_latent_sizes[m]
        with open(pfn_teacher_auc_path+f'best_pythia_pfn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            pythia_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            pythia_pfn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Pythia {a}_{b}' ,marker = '.', color = colors[n], alpha = alpha_values[m+1])
            
        with open(pfn_teacher_auc_path+f'best_herwig_pfn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            herwig_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            herwig_pfn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Herwig {a}_{b}' ,marker = 'x', color = colors[n], alpha = alpha_values[m+1])
            
        with open(pfn_teacher_auc_path+f'best_mix_pfn_latent{a}_phi{b}.txt', 'r') as file:
            lines = file.readlines()
            pythia_auc = re.findall(number_pattern, lines[0])
            mix_pfn_auc.append(float(pythia_auc[1]))
            dependency = re.findall(number_pattern, lines[2])
            mix_pfn_dependency.append(float(dependency[0]))
            
            ax.scatter([float(pythia_auc[1])], [float(dependency[0])], label = f'Mix {a}_{b}' ,marker = '*', color = colors[n], alpha = alpha_values[m+1])

ax.set_title('PFN Dependency Plot')
ax.set_xlabel('AUC on Pythia dataset')
ax.set_ylabel('AUC_Diff / AUC_on_Pythia')
ax.set_xlim(0.88,0.95)
ax.set_ylim(0,0.02)
ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(1.05,1.1))
fig.subplots_adjust(right=0.5)
plt.show()
plt.savefig(to_save_path + 'pfn_dependency_plot.jpg')

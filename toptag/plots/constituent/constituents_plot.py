import numpy as np
import matplotlib.pyplot as plt
import energyflow as ef
from energyflow.datasets import ttag_jets


# data controls, can go up to 1000000 for full Pythia dataset, 1000000 for full Herwig dataset
pythia_num = 1000000
herwig_num = 1000000
# load Pythia data
print('Loading the Pythia training dataset ...')
X_pythia, y_pythia = ttag_jets.load(pythia_num, generator='pythia')
print('Dataset loaded!')
print(X_pythia.shape)

# load Herwig data
print('Loading the Herwig training dataset ...')
X_herwig, y_herwig = ttag_jets.load(herwig_num, generator='herwig', cache_dir='~/.energyflow/herwig')
print('Dataset loaded!')
print(X_herwig.shape)

# Plot Pythia Dataset
X_pythia_qcd = [] # qcd: 0
X_pythia_top = [] # top: 1

pythia_constituent_num = []
for x in X_pythia:
    x0=x[:,0]>0
    n=x[x0].shape[0]
    pythia_constituent_num.append(n)
pythia_constituent_num = np.array(pythia_constituent_num)

pythia_qcd_constituent_num = []
pythia_top_constituent_num = []
for x in range(y_pythia.shape[0]):
    if y_pythia[x] == 0:
        pythia_qcd_constituent_num.append(pythia_constituent_num[x])
    else:
        pythia_top_constituent_num.append(pythia_constituent_num[x])
pythia_qcd_constituent_num = np.array(pythia_qcd_constituent_num)
pythia_top_constituent_num = np.array(pythia_top_constituent_num)

plt.figure()
plt.hist(pythia_constituent_num, bins=np.arange(0, 200, 2), label='Pythia Combined', alpha=0.5)
plt.hist(pythia_qcd_constituent_num, bins=np.arange(0, 200, 2), label='Pythia QCD', alpha=0.5)
plt.hist(pythia_top_constituent_num, bins=np.arange(0, 200, 2), label='Pythia Top Quark', alpha=0.5)
plt.ylabel('Number of jets')
plt.xlabel('Number of Particles')
plt.title('Pythia Dataset Constituents Count')
plt.legend()
plt.show()
plt.savefig('/users/yzhou276/work/toptag/plots/constituent/constituents_plot/pythia_constituents_count.jpg')


# Plot Pythia Dataset
X_herwig_qcd = [] # qcd: 0
X_herwig_top = [] # top: 1

herwig_constituent_num = []
for x in X_herwig:
    x0=x[:,0]>0
    n=x[x0].shape[0]
    herwig_constituent_num.append(n)
herwig_constituent_num = np.array(herwig_constituent_num)

herwig_qcd_constituent_num = []
herwig_top_constituent_num = []
for x in range(y_herwig.shape[0]):
    if y_herwig[x] == 0:
        herwig_qcd_constituent_num.append(herwig_constituent_num[x])
    else:
        herwig_top_constituent_num.append(herwig_constituent_num[x])
herwig_qcd_constituent_num = np.array(herwig_qcd_constituent_num)
herwig_top_constituent_num = np.array(herwig_top_constituent_num)

plt.figure()
plt.hist(herwig_constituent_num, bins=np.arange(0, 200, 2), label='Herwig Combined', alpha=0.5)
plt.hist(herwig_qcd_constituent_num, bins=np.arange(0, 200, 2), label='Herwig QCD', alpha=0.5)
plt.hist(herwig_top_constituent_num, bins=np.arange(0, 200, 2), label='Herwig Top Quark', alpha=0.5)
plt.ylabel('Number of jets')
plt.xlabel('Number of Particles')
plt.title('Herwig Dataset Constituents Count')
plt.legend()
plt.show()
plt.savefig('/users/yzhou276/work/toptag/plots/constituent/constituents_plot/herwig_constituents_count.jpg')

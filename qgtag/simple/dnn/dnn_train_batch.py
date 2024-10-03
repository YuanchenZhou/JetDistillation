import os

nlayers = [2,3,4,5,10]
dense_sizes = [1,10,25,50,100,200,300,500]


for n in nlayers:
    for d in dense_sizes:
        label = f'{n}layers_{d}dense_DNN'
        with open(f'batch/train_{label}.sh', 'w') as run_script:

            run_script.write('#!/bin/bash\n\n')

            run_script.write('#SBATCH -N 1\n')
            run_script.write('#SBATCH -n 1\n')
            run_script.write('#SBATCH --mem=96G\n')
            run_script.write('#SBATCH -t 12:00:00\n')

            # https://docs.ccv.brown.edu/oscar/gpu-computing/submit-gpu
            run_script.write('#SBATCH -p gpu --gres=gpu:1\n')

            run_script.write('#SBATCH -o logs/slurm-%j.out\n')
            run_script.write('#SBATCH -J DNN_'+label+'\n')
            run_script.write('\n')

            run_script.write('source /users/yzhou276/work/qgtag/qg/bin/activate\n\n')

            cmd =  'python3 dnn_train.py '
            cmd += '-nEpochs=200 '
            cmd += '-batchSize=500 '
            cmd += '-doEarlyStopping '
            cmd += '-patience=10 '
            cmd += '-usePIDs '
            cmd += '-nLayers='+str(n)+' '
            cmd += '-layerSize='+str(d)+' '
            cmd += '-ModelNum=0 '

            print(cmd)
            run_script.write(cmd+'\n')

        print('sbatch batch/train_'+label+'.sh')
        os.system(f'sbatch batch/train_{label}.sh')

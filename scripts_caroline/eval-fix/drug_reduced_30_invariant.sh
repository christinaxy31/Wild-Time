#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=64G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="drug-reduced-30-invariant-eval-fix" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=drug --method=coral --offline --lr=5e-5 --split_time=2016 --num_groups=3 --group_size=2  --coral_lambda=0.9 --mini_batch_size=256 --train_update_iter=5000 --data_dir=./Data/Drug-BA --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=drug --method=coral --offline --lr=5e-5 --split_time=2016 --num_groups=3 --group_size=2  --coral_lambda=0.9 --mini_batch_size=256 --train_update_iter=5000 --data_dir=./Data/Drug-BA --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=drug --method=coral --offline --lr=5e-5 --split_time=2016 --num_groups=3 --group_size=2  --coral_lambda=0.9 --mini_batch_size=256 --train_update_iter=5000 --data_dir=./Data/Drug-BA --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=drug --method=groupdro --offline --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --data_dir=./Data/Drug-BA --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=drug --method=groupdro --offline --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --data_dir=./Data/Drug-BA --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=drug --method=groupdro --offline --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --data_dir=./Data/Drug-BA --random_seed=3 --reduced_train_prop=0.3

python main.py --dataset=drug  --method=irm --offline --irm_lambda=1e-3 --irm_penalty_anneal_iters=0 --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --data_dir=./Data/Drug-BA --random_seed=1 --reduced_train_prop=0.3
python main.py --dataset=drug  --method=irm --offline --irm_lambda=1e-3 --irm_penalty_anneal_iters=0 --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --data_dir=./Data/Drug-BA --random_seed=2 --reduced_train_prop=0.3
python main.py --dataset=drug  --method=irm --offline --irm_lambda=1e-3 --irm_penalty_anneal_iters=0 --split_time=2016 --num_groups=3 --group_size=2 --mini_batch_size=256 --train_update_iter=5000 --lr=2e-5 --data_dir=./Data/Drug-BA --random_seed=3 --reduced_train_prop=0.3
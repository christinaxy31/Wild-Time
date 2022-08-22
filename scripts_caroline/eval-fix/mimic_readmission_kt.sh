#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G # Request 16GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --job-name="mimic-readmit-kt" # Name the job (for easier monitoring)
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cchoi1@stanford.edu     # Where to send mail

# Now your Python or general experiment/job runner code
source /iris/u/huaxiu/venvnew/bin/activate

cd ../..

python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=1 --group_size=2 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=coral --offline --prediction_type=readmission --coral_lambda=1.0 --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=groupdro --offline --prediction_type=readmission --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3

python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=1
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=2
python main.py --dataset=mimic --method=irm --offline --prediction_type=readmission --irm_lambda=1.0 --irm_penalty_anneal_iters=0 --num_groups=2 --group_size=1 --mini_batch_size=128 --train_update_iter=3000 --lr=5e-4 --num_workers=0 --split_time=2013 --random_seed=3
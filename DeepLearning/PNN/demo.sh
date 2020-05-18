python3.7 start.py --task_type=train --learning_rate=0.0005 --l2_reg=0.01 --optimizer=Adam --num_epochs=30 --batch_size=32  --dropout=0.5,0.5,0.5 --deep_layers=128,64,32 --log_steps=2000 --num_threads=8 --clear_existing_model=False
#--dist_mode=1 --job_name=${1}
#--task_index=0
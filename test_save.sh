python main.py --n_GPUs=1  \
               --data_train='RefMRI' \
               --name_train='mattrain' \
               --data_test='RefMRI' \
               --name_test='mattest' \
               --dir_data='./demodata'  \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --pre_train="./experiment/model_best.pt" \
               --test_only \
               --scale="1.5+2+3+4+6+8" \
               --model="dualref"  \
               --ref_type_test=1 \
               --ref_mat='ref_mat' \
               --ref_list='multiname.txt' \
               --save_results  \
               --savefigfilename="demo"  \


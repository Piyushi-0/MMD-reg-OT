for ktype in imq_v2
do
for khp in 10
do
for lda in 1
do
for lr in 0.001
do
for alpha in 0.01 #same as default
do
for lambda_reg in 0.5 #same as default
do
python3 train_jumbot.py UOT --gpu_id [0] --lr $lr --alpha $alpha --lambda_reg $lambda_reg --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Art.txt --t_dset_path ./data/office-home/Clipart.txt --batch_size 65 --output_dir "A_C" --ktype ${ktype} --khp ${khp} --lda ${lda}
done
done
done
done
done
done
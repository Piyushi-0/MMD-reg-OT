GPU_IDX=$1

for n in 40 60 80 100 200 300 400 500 1000
do
python emd.py --n ${n} --gpu_idx ${GPU_IDX} >> emd.txt
done

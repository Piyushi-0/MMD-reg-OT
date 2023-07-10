GPU_IDX=$1

for n in 40 60 80 100 200 300 400 500 1000
do
python w2.py --n ${n} --gpu_idx ${GPU_IDX} >> w2.txt
done

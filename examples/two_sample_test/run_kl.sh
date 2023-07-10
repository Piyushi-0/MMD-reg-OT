GPU_IDX=$1

for n in 20 40 60 80 100 200 300 400 500 1000
do
python kl.py --n ${n} > "kl_${n}.txt" --gpu_idx ${GPU_IDX} >> klot.txt
done

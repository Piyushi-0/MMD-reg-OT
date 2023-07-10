GPU_IDX=$1

for n in 40 60 80 100 200 300 400 500 1000
do
python mmdot.py --max_iter 100 --ktype rbf --case unb --n ${n} --p 2 --gpu_idx ${GPU_IDX} >> mmdot.txt
done

for n in 100 200 300 400 500 1000
do
python eps_ot.py --n ${n} --gpu_idx 0 >> ot.txt
done

cd "$(dirname "$0")/.."  # cd to repo root.

mkdir -p results
mkdir -p model
device=c  # For CPU set device=c; for CUDA set device=0 (if 0 is your the CUDA device number)



############################## New dataset dose_exp ###############################

echo "Hybrid"
sample=310
model_path="model/model_sample_${sample}/"
python -u -m experiments.run_simulation --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 "$@" > "results/sample_${sample}_h.txt" || echo Error!!!


sample_arr=( 400 800 )

for sample in "${sample_arr[@]}"
do
    echo "$sample"
    model_path="model/model_sample_${sample}/"
    python -u -m experiments.run_simulation --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 "$@" > "results/sample_${sample}_h.txt" || echo Error!!!
done


######################## Evaluation #################################

echo Evaluation
data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid )
sample_arr=( 310 400 800 )

for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        if [ $m = expert ] ; then
          flagadd="$@"
        else
          flagadd=
        fi
        python -u -m experiments.run_simulation --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y $flagadd > "results/sample_${sample}_${m}.txt" &
    done
done

for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    python -u -m experiments.run_simulation_ensemble --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2.txt"
    python -u -m experiments.run_simulation_residual --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_residual.txt"
done



######################## Summary #################################

model_arr=( hybrid )
seed_arr=( 310 400 800 )

rm -f results/results_sample.txt
for seed in "${seed_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        value=`tail -n 4 results/sample_${seed}_${m}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${m},${seed},${line}" >> results/results_sample.txt
        done
    done
done

grep rmse_x results/results_sample.txt

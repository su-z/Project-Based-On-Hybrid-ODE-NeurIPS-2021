cd "$(dirname "$0")/.."  # cd to repo root.

device=c  # For CPU set device=c; for CUDA set device=0 (if 0 is your the CUDA device number)
sample=400
sample_total=`expr ${sample} + 800`



############################## New dataset dose_exp ###############################

noise_arr=( 0.4 0.8 )

for noise in "${noise_arr[@]}"
do
    model_path="model/model_noise_${noise}/"
    data_path="data/datafile_dose_noise_${noise}.pkl"

    model_arr=( neural expert hybrid )
    for method in "${model_arr[@]}"
    do
        m=$method
        if [ $m = expert ] ; then
          flagadd="$@"
        else
          flagadd=
        fi
        python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --batch_size=10 --path=${model_path} --data_path=${data_path} $flagadd > "results/noise_${noise}_${method}.txt"
    done

    python -u -m experiments.run_simulation_flow --method=hybrid --device=${device} --sample=${sample_total} --batch_size=10 --path=${model_path} --data_path=${data_path} > "results/noise_${noise}_flow.txt"

    model_path="model/model_noise_${noise}/"
    sample_cali=`expr ${sample} - 300`
    python -u -m experiments.run_simulation_ensemble --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/noise_${noise}_ensemble2.txt"
    python -u -m experiments.run_simulation_residual --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/noise_${noise}_residual.txt"

done

model_arr=( neural expert hybrid residual ensemble2 flow )
seed_arr=( 0.4 0.8 )

rm -f results/results_noise.txt
for seed in "${seed_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        value=`tail -n 4 results/noise_${seed}_${m}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${m},${seed},${line}" >> results/results_noise.txt
        done
    done
done

sample=400
for m in "${model_arr[@]}"
do
    value=`tail -n 4 results/sample_${sample}_${m}.txt`
    readarray -t y <<<"$value"
    for line in "${y[@]}"
    do
        echo "${m},0.2,${line}" >> results/results_noise.txt
    done
done

grep rmse_x results/results_noise.txt

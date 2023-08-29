function grid_search() {
file=$1
action_command=$3
arguments_action_command=$4
workers=$2
while read line; do
    echo "Start law run for model ${line}"
  law run cf.MLTraining --version 1 --ml-model $line --config run2_2017_nano_uhh_v11_limited --workers $workers $action_command $arguments_action_command
done < "${file}"
}

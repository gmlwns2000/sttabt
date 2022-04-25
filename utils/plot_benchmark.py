import matplotlib.pyplot as plt
import sys
import subprocess
import pickle
import os

exp_name = ""
exp_device = 'cuda'
exp_amp = False
exp_skip_accuracy = False
exp_batch_size = {'bert-mini':736, 'bert-base':128}

def run_exp(model, factor, dropout, target='sparse'):
    global exp_device, exp_amp, exp_skip_accuracy, exp_batch_size
    batch_size = exp_batch_size[model]
    cmd = f"python -m main.benchmark_approx"
    cmd = cmd + f" --batch-size {batch_size}"
    cmd = cmd + f" --factor {factor}"
    cmd = cmd + f" --target {target}"
    cmd = cmd + f" --model {model}"
    cmd = cmd + f" --dropout {dropout}"
    cmd = cmd + f" --device {exp_device}"
    if exp_skip_accuracy: cmd = cmd + " --skip-accuracy"
    if exp_amp: cmd = cmd + " --amp"
    print(cmd)
    subprocess.call(cmd.split(' '))
    with open('bench_result.pkl', 'rb') as f:
        obj = pickle.load(f)
        speed, acc = obj
    os.remove('bench_result.pkl')
    print('done', speed, acc)
    return speed, acc

def plot_exp_by_factor(model, factor):
    dropouts = [0.1, 0.25, 0.5, 0.75, 0.99]
    speedups = []
    accuracies = []
    bert_speed, bert_acc = run_exp(model, factor, 0.5, 'bert')
    for dropout in dropouts:
        speed, acc = run_exp(model, factor, dropout)
        speedups.append(speed / bert_speed)
        accuracies.append(acc / bert_acc)
    return dropouts, speedups, accuracies

def plot_exp(model='bert-mini'):
    global exp_device, exp_amp, exp_skip_accuracy, exp_batch_size, exp_name

    plt.style.use("seaborn")

    d, s4, a4 = plot_exp_by_factor(model, 4)
    d, s8, a8 = plot_exp_by_factor(model, 8)
    d, s16, a16 = plot_exp_by_factor(model, 16)

    print('speedup')
    plt.plot(d, s4, label='factor:4')
    plt.plot(d, s8, label='factor:8')
    plt.plot(d, s16, label='factor:16')
    plt.legend()
    plt.savefig(f'saves_plot/{exp_name}{model}_dev_{exp_device}_amp_{exp_amp}_bsize_{exp_batch_size[model]}_speedup.png', dpi=320)
    plt.show()

    print('acc reproduce')
    plt.plot(d, a4, label='factor:4')
    plt.plot(d, a8, label='factor:8')
    plt.plot(d, a16, label='factor:16')
    plt.legend()
    plt.savefig(f'saves_plot/{exp_name}{model}_dev_{exp_device}_amp_{exp_amp}_bsize_{exp_batch_size[model]}_acc.png', dpi=320)
    plt.show()

    return [(d, s4, a4), (d, s8, a8), (d, s16, a16)]

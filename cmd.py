
import itertools
import os

# 定义超参数
dropouts = [0.1, 0.2, 0.3, 0.4, 0.5]
epochs = [30, 50, 100]
hidden_sizes = [256, 128]
out_size=[32,64]
heads = [1, 2, 4, 8]

# 获取超参数的笛卡尔积
combinations = list(itertools.product(dropouts, epochs, hidden_sizes, out_size, heads))

command_counter = 1
for combination in combinations:
    dropout, epoch, hidden_size, out_ch, head = combination
    command = f'python main.py --dropout {dropout} --EPOCHS {epoch} --HIDDEN-SIZE {hidden_size} --OUT-CHANNELS {out_ch} --HEADS {head} --output-path test_out/her/{command_counter}/'
    print("运行命令：", command)
    os.system(command)
    command_counter += 1
    print("command_counter:")
    print(command_counter)
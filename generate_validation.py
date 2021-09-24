import shutil
import random
from glob import glob

random.seed(20210830)
zeros = glob('./dataset/train/0/*.jpg')
ones = glob('./dataset/train/1/*.jpg')
zeros_valid = random.sample(zeros, 200)
ones_valid = random.sample(ones, 200)
for zero, one in zip(zeros_valid, ones_valid):
    zeros.remove(zero)
    print('deleting ' + zero)
    ones.remove(one)
    print('deleting ' + one)
    shutil.copy(zero, './dataset/train/valid/0')
    print('copying  ' + zero)
    shutil.copy(one, './dataset/train/valid/1')
    print('copying  ' + one)

for zero in zeros:
    shutil.copy(zero, './dataset/train/train/0')
    print('copying  ' + zero)

for one in ones:
    shutil.copy(one, './dataset/train/train/1')
    print('copying  ' + one)

print('done')

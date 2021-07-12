import numpy
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# seed 값 설정
# seed = 0
# numpy.random.seed(seed)
# tf.random.set_seed(3)

df = pd.read_csv("./data/data_frame_v1.csv")

print(df.info())
print(df.head())
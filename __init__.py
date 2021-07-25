import pandas as pd

H = W = 28

label_map = pd.read_csv("images/emnist/emnist-balanced-mapping.txt", 
                        delimiter = ' ', 
                        index_col=0, 
                        header=None, 
                        squeeze=True)
label_map = {k: chr(v) for k, v in label_map.items()}
allowed_chars = '12345678KQNRBXabcdefgh'
allowed_chars = list(filter(lambda k: label_map[k] in allowed_chars, label_map.keys()))

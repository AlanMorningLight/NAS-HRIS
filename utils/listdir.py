import os

pathDir = os.listdir('L:\Dataset\Land-use datasetGID\GID_dataset_2048\image')
i = 0
for name in pathDir :
    na = name[0:20]
    if na in name :

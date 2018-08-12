import matplotlib.pyplot as plt
import numpy as np

data_path = '/gpfs/data_vector/pdtw24/PL_Data/RJLens4/Object_Files/RJLensPy.dat'

file = open(data_path, 'r')

fileread = file.readlines()

file_path = '/gpfs/data_vector/pdtw24/PL_Data/RJLens4/Object_Files/RJLens.dat'

image = np.zeros((422,422))

with open(file_path, "w+") as f:
    for i, lineread in enumerate(fileread):
        line = str(round(float(fileread[i][0:13]), 2))
        line += ' ' * (8 - len(line))
        line += str(round(float(fileread[i][13:25]), 2))
        line += ' ' * (16 - len(line))
        line += str(float(fileread[i][25:80])) + '\n'
        image[int(fileread[i][0:13]), int(fileread[i][13:25])] = float(fileread[i][25:80])
        print(int(fileread[i][0:13]), int(fileread[i][13:25]), float(fileread[i][25:80]))
        f.write(line)

plt.imshow(image)
plt.show()
data_path = '/gpfs/data/pdtw24/PL_Data/RJLens4/Object_Files/RJLens.dat'

file = open(data_path, 'r')

fileread = file.readlines()


file_path = '/gpfs/data/pdtw24/PL_Data/RJLens4/Object_Files/RJLensPy.dat'

with open(file_path, "w+") as f:
    for i, lineread in enumerate(fileread):
        line = str(round(float(fileread[i][0:13]), 2))
        line += ' ' * (8 - len(line))
        line += str(round(float(fileread[i][13:25]), 2))
        line += ' ' * (16 - len(line))
        line += str(float(fileread[i][25:39])) + '\n'
        f.write(line)
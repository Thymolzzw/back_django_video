import numpy as np

file = "D:\TensorFlow_workplace_new\demo_django\statics\\resource\\face_npy\\res_list.npy"
toFile = "D:\TensorFlow_workplace_new\demo_django\statics\\resource\\face_npy\\res_list_new.npy"

array = np.load(file, allow_pickle=True)

print(array)
for item in array:
    print(item[1])
    for i in range(len(item[1])):
        item[1][i] = int(item[1][i]/29.97)
        print(item[1][i])
np.save(toFile, array)

array = np.load(toFile, allow_pickle=True)
print(array)

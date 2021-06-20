import os
import random
import string

file_dir = os.path.dirname(os.path.realpath('__file__'))
print(file_dir)
write_file_dir = os.path.join(file_dir, '..\data\data_write.txt')
print(write_file_dir)
result_file_dir = os.path.join(file_dir, '..\data\data_result.txt')
print(result_file_dir)


write_file = open(write_file_dir)
result_file = open(result_file_dir)

write_data = write_file.read().split("\n")
result_data = result_file.read().split("\n")

matched = 0
failed = 0

for i in range(len(write_data)-1):
    if(write_data[i] == result_data[i]):
        matched = matched + 1
    else:
        failed = failed + 1
        print("Data mismatch on index " + str(i))
        print("Write: " + write_data[i])
        print("Read: " + result_data[i])
        print("\n")

print("Number of matched items: " + str(matched))
print("Number of failed items: " + str(failed))

pass_rate = matched / (matched + failed) * 100
pass_rate = round(pass_rate,4)
print("\nPass rate: " + str(pass_rate) + "%")

write_file.close()
result_file.close()

import csv
lst = []
dict = {'comment_sad':0.23, 'comment_fear': 0.28}
for i in range(3):
    lst.append(dict)
print(lst)

path = "C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\test.csv"
with open(path, newline="", mode='w') as f:
    writer = csv.writer(f)
    keys = lst[0].keys()
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(lst)
    f.close()
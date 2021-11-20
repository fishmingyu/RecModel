import csv
UserDict = dict()
ItemDict = dict()
UserItemDict = dict()
# f1 = open('graph.txt', 'w')

# f1.write("%%MatrixMarket matrix coordinate integer general")
file_path = "/work/shared/common/project_build/gnn-optane/data/all_csv_files.csv"
with open(file_path) as f:
    reader = csv.reader(f)
    users = 0
    items = 0
    usertmp = ''
    for row in reader:
        if row[0] not in UserDict.keys():
            UserItemDict[row[0]] = set()
            UserDict[row[0]] = users
            users += 1
        if row[1] not in ItemDict.keys():
            ItemDict[row[1]] = items
            items += 1
        UserItemDict[row[0]].add(ItemDict[row[1]])
        # f1.write(str(UserDict[row[0]]) + ' ' + str(ItemDict[row[1]]) + ' ' + str(int(float(row[2]))) + '\n')

f2 = open('user_item_list.txt', 'w')
for key in UserItemDict:
    f2.write(str(UserDict[key]) + ' ')
    item_list = list(UserItemDict[key])
    for i in item_list:
        if i is item_list[len(item_list) - 1]:
            f2.write(str(i) + '\n')
        else:
            f2.write(str(i) + ' ')
# f1.close()
f2.close()
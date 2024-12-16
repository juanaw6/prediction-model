def get_sublists_from_back(lst, min_length=3, max_length=5):
    return [lst[i:] for i in range(len(lst) - min_length, -1, -1) if max_length is None or len(lst[i:]) <= max_length]

def get_sublists(lst, min_length=3, max_length=5):
    sublists = [lst[:i+min_length] for i in range(max_length - min_length + 1)]
    return sublists

test_list = [x for x in range(10)]
print(test_list)
result = get_sublists_from_back(test_list)

for l in result:
    print(l)

test_list.reverse()
print()
print(test_list)
result = get_sublists(test_list)

for l in result:
    print(l)
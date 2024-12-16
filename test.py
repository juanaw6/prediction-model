def get_sublists(lst, min_length=3, max_length=5):
    sublists = [lst[:i+min_length] for i in range(max_length - min_length + 1)]
    return sublists

test_list = [x for x in range(10)]

test_list.reverse()

print(test_list)

result = get_sublists(test_list)

for l in result:
    print(l)
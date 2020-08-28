import operator
d = {1:2,2:4}
m = (min(d.items(), key=operator.itemgetter(1))[0])
print('min rmse{} - k{}'.format(d[m],m))



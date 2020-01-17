with open('test4.json', 'r') as fin:
    df = {}
    i = 0
    for line in fin:
        df[i] = eval(line)
        print("第%d条" % i)
        print(df[i])
        i += 1
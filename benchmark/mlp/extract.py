with open('mem-info.log', 'r') as log_reader:
    info_list = log_reader.readlines()
    for info in info_list:
        info = info.rstrip('\n')
        it = info.split(' ')
        if it[0] == 'READ' or it[0] == 'WRITE':
            if int(it[3]) != 0:
                print('===> {}'.format(info))


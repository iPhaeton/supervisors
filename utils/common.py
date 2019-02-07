def filter_list(refs, condition, l):
    if condition == True:
        return list(filter(lambda x: x in refs, l))
    else:
        return list(filter(lambda x: x not in refs, l))

'''Create a phone root to dependant phone mapping'''

import argparse

def read_root(fid):
    dict_map = {}
    with open(fid, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 7:
                root = tokens[2]
            else:
                root = tokens[2].split('_')[0]
            for i in tokens[2:]:
                dict_map[i] = root
    return dict_map

def read_phones(fid):
    dict_phones = {}
    stars = []
    eps = ''
    with open(fid, 'r') as f:
        for line in f:
            if 'eps' in line:
                eps = line.strip()
            elif '#' in line:
                stars.append(line.strip())
            else:
                tokens = line.strip().split()
                dict_phones[tokens[0]] = tokens[1]
    return dict_phones, eps, stars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='data/lang/phones/root.txt')
    parser.add_argument('phones', help='data/lang/phones.txt')
    parser.add_argument('output', help='outputfile')
    args = parser.parse_args()

    root = read_root(args.root)
    phone_map, eps, stars = read_phones(args.phones)

    dict_root = {}
    with open(args.output, 'w') as f:
        print(eps, file=f)
        for i in phone_map.keys():
            phone_id = phone_map[i]
            print(root[i], phone_id, file=f)
        for i in stars:
            print(i, file=f)

if __name__ == '__main__':
    main()



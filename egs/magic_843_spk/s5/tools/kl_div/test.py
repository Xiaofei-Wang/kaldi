import argparse
import post_io

def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('post')
    parser.add_argument('num_phones')
    args = parser.parse_args()

    y = post_io.readFileHead(args.post, args.num_phones)
    for k in y.keys():
        list_k = y[k].tolist()
        print (list_k)

if __name__ == '__main__':
    main()
else:
    raise ImportError('Can not be imported')



if __name__ == '__main__':

parser = argparse.ArgumentParser(description='PyTorch Example for model comporession')

parser.add_argument('--dataset', type=str, default='mnist',help='dataset')

parser.add_argument()

parser.add_argument()

......

args = parser.parse_args()

main(args)
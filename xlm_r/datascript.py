import argparse
file_english_a = "data/task_a_distant.tsv"

parser = argparse.ArgumentParser(description='Process data according to specification')
parser.add_argument('--size', type=int, nargs=1, default=25000)
parser.add_argument('--lang', type=str, nargs='+', default=["English"])
parser.add_argument('--task', type=str, nargs=1, default='a')
parser.add_argument('--out', type=str, nargs=1, default="data/train.tsv")

args = parser.parse_args()


class ReadTweet:
    def __init__(self):
        self.output_file = args.out
        self.size = 0
        self.max_size = args.size

    def parse_en(self, output_file):
        with open(file_english_a, encoding="utf-8") as f:
            f.readline()  # Skip first line of table headers
            for line in f:
                current = line.rstrip()
                current = current.split("\t")
                tweet = str(current[1])
                conf = float(current[2])
                if conf > 0.5:
                    conf = 1
                else:
                    conf = 0
                out = str(tweet) + "\t" + str(conf) + "\n"
                output_file.write(out)
                self.size += 1
        print(self.size)
                # if self.size >= self.max_size:
                #     break


if __name__ == '__main__':
    output_file = open(args.out, 'wt', encoding="utf-8")
    print(args)
    p = ReadTweet()
    p.parse_en(output_file)
    output_file.close()

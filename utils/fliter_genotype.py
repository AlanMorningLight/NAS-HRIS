import os

def main():
    result = open('./result.txt','a')
    result.seek(0)
    result.truncate()
    with open(r'L:\NAS\NAS-RSI1\log.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'train_acc' in line:
                result.write(line)
    result.close()

if __name__ == '__main__':
    main()
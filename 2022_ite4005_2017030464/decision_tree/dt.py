import sys

def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    min_sup = int(args[0])
    try:
        input = open(args[1], 'r', encoding='utf-8')
    except FileNotFoundError:
        print("*** input 파일이 없습니다. ***\n")
        return 0
    output = open(args[2], 'w', encoding='utf-8')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
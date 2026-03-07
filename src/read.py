import json
from file_handler import file_read


def main():
    path = input("Enter file path: ").strip()

    count = 0

    for line in file_read(path):
        msg = json.loads(line)

        print(msg)
        count += 1

    print(f"\nread {count} messages")


if __name__ == "__main__":
    main()
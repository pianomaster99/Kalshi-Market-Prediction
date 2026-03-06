from file_handler import file_read

for obj in file_read("data/0305.ndjson.gz"):
    print(obj)
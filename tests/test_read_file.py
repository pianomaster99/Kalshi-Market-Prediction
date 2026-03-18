from file_handler import file_read

i = 0
for obj in file_read("data/KXNBAGAME-26MAR12BOSOKC-BOS.ndjson.gz"):
    i += 1
    if i > 20:
        break
    print(obj)
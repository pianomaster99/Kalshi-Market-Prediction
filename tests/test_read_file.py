from data_collection.file_handler import file_read

i = 0
for obj in file_read("data/raw_data/KXNBAGAME-26MAR18ATLDAL-DAL.ndjson.gz"):
    i += 1
    if i > 20:
        break
    print(obj)
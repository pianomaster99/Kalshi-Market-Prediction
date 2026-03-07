from file_handler import file_read

for obj in file_read("data/KXATPCHALLENGERMATCH-26MAR07SAMSAK-SAM.ndjson.gz"):
    print(obj)
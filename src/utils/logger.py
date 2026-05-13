import csv
import os


class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        self.file = open(filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()
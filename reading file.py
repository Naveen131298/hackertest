import os
import glob


def read_pdf(path):
    for pdf_file in glob.glob(path + '/*.pdf'):
        print(pdf_file)


pdf_location = os.path.join(os.getcwd())
read_pdf(pdf_location)
print(os.path.join(os.getcwd()))

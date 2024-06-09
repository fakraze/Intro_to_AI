import zipfile
with zipfile.ZipFile("./mahjong_data/data.zip","r") as zip_ref:
    zip_ref.extractall("./mahjong_data/data")
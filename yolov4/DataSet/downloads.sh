# !/bin/sh
CLASS=Coin
TYPE=all # validation, train, test
git clone https://github.com/EscVM/OIDv4_ToolKit.git
cd OIDv4_ToolKit
pip3 install -r requirements.txt
python3 main.py downloader --classes ${CLASS} --type_csv ${TYPE} --yes
mv OID/Dataset ../${CLASS}

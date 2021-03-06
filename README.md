# Yellot project

## Installation

Install system requirements
```bash
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-rus
sudo apt-get install libsm6 libxrender1 libfontconfig1
sudo apt-get install poppler-utils
sudo apt-get install python3-pip
sudo apt install openjdk-11-jdk # for svgNest
```

Torch installation may be finished with MemoryError. Then
```
wget https://files.pythonhosted.org/packages/46/ca/306bb933a68b888ab1c20ede0342506b85857635f04fb55a56e53065579b/torch-1.4.0-cp27-cp27mu-manylinux1_x86_64.whl
sudo pip3 install torch-1.4.0-cp27-cp27mu-manylinux1_x86_64.whl
```


Install other python dependencies
```
sudo pip3 install setuptools
sudo pip3 install -r requirements.txt
```

Install tesseract languages data
```
cp sevice/text_recog/tunedeng.traineddata /usr/share/tesseract-ocr/5/tessdata/
```

## Run


Run the server 
```
cd service
python3 server.py 
```


## Proxy

On test server [ngrok](https://dashboard.ngrok.com/get-started) had been used.
```
ngrok http 5022 
```

then tunnel 5022 -> 80 will start. Foreign address can be viewed on ngrok web dashboard

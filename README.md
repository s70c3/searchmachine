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

## Run

Replace last string in server.py with your server ip and preferable port
```
app.run(debug=False, host='your_ip', port=your_port)
```

Run the server 
```
python3 server.py 
```


## Proxy

On test server [ngrok](https://dashboard.ngrok.com/get-started) had been used.
```
ngrok http 5022 
```

then tunnel 5022 -> 80 will start. Foreign address can be viewed on ngrok web dashboard

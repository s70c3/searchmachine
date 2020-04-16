# Yellot project

## Installation

Install system requirements
```bash
sudo add-apt-repository ppa:alex-p/tesseract-ocr
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-rus
sudo apt-get install python3-pip
```

Install python dependencies
```
sudo pip3 install -r requirements.txt
```

## Run

Server starts on port 5022 
```
python3 server.py 
```


## Proxy

On test server [ngrok](https://dashboard.ngrok.com/get-started) had been used.
```
ngrok http 5022 
```

then tunnel 5022 -> 80 will start. Foreign address can be viewed on ngrok web dashboard
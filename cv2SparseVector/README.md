# vector (image) to sparse ones by opencv and kmeans

Extract main color, quantize its vector and let its entropy be less .

it can be used to preprocesing for DNN and son on.


<img src="https://user-images.githubusercontent.com/48679574/228387708-599da600-9ff2-4029-8cf9-5d64ef6c3f08.png" width="500" height="300"/>



## How to use

```sh
# sparse 
python3 main.py --input_path "data/input_lenna.png" --output_name "data/output_lenna"

# get sparse more smooth
$ python3 main --smooth

# sparse and plot result
$ python3 main --plot
```


## Overall flow & Relations of entropy and kluster

#### Overall flow

<img src="https://user-images.githubusercontent.com/48679574/228399674-bf251f33-4f9c-4a96-80e3-56897e5d9f4f.jpg" width="500" height="300"/>

#### Rerations of entropy and kluster

<img width="497" alt="kluster_entropy" src="https://user-images.githubusercontent.com/48679574/228390072-7bdce12e-fd44-4a86-8484-442b0e6a786e.png">


# Performance

<b>input(kluster=256) / output(kluster=3)</b>

<img src="https://user-images.githubusercontent.com/48679574/228390397-80fb80da-0a29-43ca-b95d-a768152e5ffe.png" width="200" height="200"/><img src="https://user-images.githubusercontent.com/48679574/228390403-7f80cb00-b7f3-4179-b1fd-b8ac7d290f8f.png" width="200" height="200"/>


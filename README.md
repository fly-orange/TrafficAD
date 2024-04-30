# TrafficAD
This is the code of "Towards Efficient Traffic Incident Detection via Explicit Edge-Level Incident Modeling" (IEEE IoT Journal 2024)

## Data
Original datasets can be downloaded at https://drive.google.com/file/d/1vgeBM5sYZq75ziiyOVC9ypIkKVVztG3W/view?usp=drive_link

1. Download the two datasets in bay/data and la/data, respectively.

2. Preprocess original data to get asy_graph.npy, event_label_v.npy, and event_label_e.npy

## Model

‘python main.py --dataset dataset_name --detector model_name’

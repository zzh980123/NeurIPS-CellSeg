sudo docker container run --gpus "device=2" -m 28G --name naf --rm -v /media/kevin/870A38D039F26F71/Datasets/NeurISP2022-CellSeg/TuningSet/:/workspace/inputs/ -v /media/kevin/870A38D039F26F71/PycharmProjects/CellSeg/NeurIPS-CellSeg/dev_workspace/outputs/:/workspace/outputs/ naf:latest /bin/bash -c "sh predict.sh"

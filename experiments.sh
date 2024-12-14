ID=0
BATCH_SIZE=256

python3 main.py --run-id=$ID --lr=0.00001 --conv=SAGEConv  --decoder=LinearDecoder     --hidden-layers=0 --batch-size=$BATCH_SIZE --epochs=50
python3 main.py --run-id=$ID --lr=0.00001 --conv=SAGEConv  --decoder=LinearDecoder     --hidden-layers=2 --batch-size=$BATCH_SIZE --epochs=50
python3 main.py --run-id=$ID --lr=0.00001 --conv=SAGEConv  --decoder=DotProductDecoder --hidden-layers=0 --batch-size=$BATCH_SIZE --epochs=100
python3 main.py --run-id=$ID --lr=0.00001 --conv=SAGEConv  --decoder=DotProductDecoder --hidden-layers=2 --batch-size=$BATCH_SIZE --epochs=100

python3 main.py --run-id=$ID --lr=0.00001 --conv=GraphConv --decoder=LinearDecoder --hidden-layers=0 --batch-size=$BATCH_SIZE --epochs=50
python3 main.py --run-id=$ID --lr=0.00001 --conv=GraphConv --decoder=LinearDecoder --hidden-layers=2 --batch-size=$BATCH_SIZE --epochs=50
python3 main.py --run-id=$ID --lr=0.00001 --conv=GraphConv --decoder=DotProductDecoder --hidden-layers=0 --batch-size=$BATCH_SIZE --epochs=100
python3 main.py --run-id=$ID --lr=0.00001 --conv=GraphConv --decoder=DotProductDecoder --hidden-layers=2 --batch-size=$BATCH_SIZE --epochs=100

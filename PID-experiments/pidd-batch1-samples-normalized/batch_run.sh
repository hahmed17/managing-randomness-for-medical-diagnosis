i=1
while [ $i -le 100 ]
do
    mkdir sample$i
    cd sample$i
    python3 ..\..\neural-net.py 1
    cd ..
done
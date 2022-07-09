for /l %%i in (1, 1, 719) do (
    mkdir sample%%i
    cd sample%%i
    python3 ..\..\neural-net.py %%i
    cd ..
)
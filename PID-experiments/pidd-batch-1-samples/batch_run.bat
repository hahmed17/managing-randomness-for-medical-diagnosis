for /l %%i in (18, 1, 100) do (
    mkdir sample%%i
    cd sample%%i
    python3 ..\..\neural-net.py
    cd ..
)
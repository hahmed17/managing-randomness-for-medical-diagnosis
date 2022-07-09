for /l %%i in (1, 1, 100) do (
    mkdir sample%%i
    cd sample%%i
    python3 ..\..\models.py gaussian-naive-bayes ..\..\datasets\heart-disease.csv 
    cd ..
)
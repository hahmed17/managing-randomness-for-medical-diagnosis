for /l %%i in (1, 1, 100) do (
    mkdir sample%%i
    cd sample%%i
    python3 ..\..\models.py bernoulli-naive-bayes ..\..\datasets\compiled_heart_disease.csv 
    cd ..
)
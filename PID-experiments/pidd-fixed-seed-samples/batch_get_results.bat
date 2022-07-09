for /l %%i in (1, 1, 719) do (
    cd sample%%i
    awk /Accuracy/ out.txt >> ../accuracy.txt
    cd ..
)
cut -d " " -f 2- accuracy.txt > diabetes-results-varied-batch.txt
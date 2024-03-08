#/bin/bash

cd /home/hzhao/sample
python -c "import deleteDays; deleteDays.deleteDays($1,$2,$3)"
echo "Model results on irrigated days are removed."
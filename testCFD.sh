#!/bin/bash
source /home/hzhao/.bashrc
source /home/hzhao/anaconda3/bin/activate WS

echo "Creating forcing data..."
cd /home/hzhao/sample
python -c "import create_forcing_data; create_forcing_data.createForcingData($1,$2,$3,$4,$5)"
echo "Finished!"

#python /home/hzhao/sample/create_setup_file.py
python -c "import create_setup_file; create_setup_file.CreateDat($6,$7,$3,$1,$2)"
echo "Bondville file created!
"
cd /home/hzhao/single-point/workshop
./create_point_data.exe
echo "Setup file created!"

cd /home/hzhao/sample
python -c "import create_namelist; create_namelist.createFile($1,$2,$3,$4)"
echo "Namelist created!"

cd /home/hzhao/single-point/newrun/hrldas/run
./hrldas.exe
echo "Model run finished!"

cd /home/hzhao/sample
python -c "import irriTable; irriTable.irriTable($3,$4,'$8',$5,$1,$2,$9,${10},'${11}',$6,$7,${12},'${13}')"
echo "Irrigation table generated!"

python -c "import yieldEstimate; yieldEstimate.yield_est($7,$6,$9,${10},$3,$4,$5,$1,$2,'${11}',${12},'${13}')"
echo "Yield estimations generated!"
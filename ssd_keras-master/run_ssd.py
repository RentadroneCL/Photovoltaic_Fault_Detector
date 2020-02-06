import os
#os.mkdir('../Experimento_5/Resultados_ssd')
#os.mkdir('../Experimento_3/Resultados_ssd/ssd512')
#os.mkdir('../Experimento_5/Resultados_ssd/ssd300')
#os.mkdir('../Experimento_3/Resultados_ssd/ssd7')

#print ('Training ssd7')
#os.system('python train.py -c config_7.json > ../Experimento_3/Resultados_ssd/ssd7/ssd7.output 2> ../Experimento_3/Resultados_ssd/ssd7/ssd7.err') 

#print ('Testing ssd7')
#os.system('python evaluate.py -c config_7.json > ../Experimento_3/Resultados_ssd/ssd7/ssd7_test.output 2> ../Experimento_3/Resultados_ssd/ssd7/ssd7_test.err')


print ('Training ssd300')
os.system('python train.py -c config_300.json > ../Experimento_5/Resultados_ssd/ssd300/ssd300.output 2> ../Experimento_5/Resultados_ssd/ssd300/ssd300.err')
print ('Testing ssd300')
os.system('python evaluate.py -c config_300.json > ../Experimento_5/Resultados_ssd/ssd300/ssd300_test.output 2> ../Experimento_5/Resultados_ssd/ssd300/ssd300_test.err')

#print ('Training ssd512')
#os.system('python train.py -c config_512.json > ../Experimento_3/Resultados_ssd/ssd512/ssd_512.output 2> ../Experimento_3/Resultados_ssd/ssd512/ssd_512.err')
#print ('Testing ssd7')
#os.system('python evaluate.py -c config_512.json > ../Experimento_3/Resultados_ssd/ssd512/ssd512_test.output 2> ../Experimento_3/Resultados_ssd/ssd512/ssd512_test.err')






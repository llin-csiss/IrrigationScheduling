
#! /usr/bin/python

import sys, time
import os

def CreateLDASIN(input, output, x, y, rainvalue):
    #os.system("ncks -d south_north," + y + "," + y + " -d west_east," + x + "," + x + " " + input + " " + output)
    os.system("ncks -d south_north,"+str(int(y))+","+str(int(y))+" -d west_east,"+str(int(x))+","+str(int(x))+" "+input+" "+output)
    os.system("ncap2 -O -s 'rainrate_nwm=rainrate_nwm+"+ str(rainvalue)+"' "+output+" " +output)
    os.system("ncrename -v t2d_nwm,T2D " +output)
    os.system("ncrename -v q2d_nwm,Q2D " +output)
    os.system("ncrename -v psfc_nwm,PSFC " +output)
    os.system("ncrename -v u2d_nwm,U2D " +output)
    os.system("ncrename -v v2d_nwm,V2D " +output)
    os.system("ncrename -v lwdown_nwm,LWDOWN " +output)
    os.system("ncrename -v swdown_nwm,SWDOWN " +output)
    os.system("ncrename -v rainrate_nwm,RAINRATE " +output)
    os.system("chmod +x "+ output)
    #os.system("rm temp2.nc")

def GetFiles(inPath, outPath, start, end, irri_record):
    if not os.path.isdir(inPath):
        print('Error: Path is not a directory!')
        return
    entries = os.listdir(inPath)
    entries = [entry for entry in entries if entry.find('RESTART') == -1]
    param = []
    #param.append(['/home/mbarlage/data/output/analysis/RESTART.2020040100_DOMAIN1', outPath+'/RESTART.2020040100_DOMAIN1'])
    for entry in entries:
        rainvalue = 0
        changed = False
        if int(entry.split('.')[0]) >= start and int(entry.split('.')[0]) <= end:
            if len(irri_record)>0:
                for item in irri_record:
                    date = item["start"].split('T')[0].replace('-','')
                    if entry.find(str(date)) != -1:
                        changed = item["changed"]
                        if changed == 'false': changed = False
                        if changed == 'true': changed = True
                        rainvalue = rainvalue + (float(item["volume"])/(24*3600) if float(item["volume"])>=0 else 0)
                        break
            outEn = entry
            param.append([inPath+'/'+entry,outPath+'/'+outEn, rainvalue, changed])
    return param

def createForcingData(x,y, start, end,irri_record):

    NWMPath = '/home/mbarlage/data/LDASIN/analysis'
    outPathX = r'/home/hzhao/single-point/analysis/'+str(int(x))
    outPathY = outPathX + '/' + str(int(y))
    #nonIrrPath = r'/home/hzhao/single-point/analysis/analysis'+site
    irrPath = '/home/hzhao/sample/record.csv'

    if not os.path.isdir(outPathX):
        os.makedirs(outPathX)
    if not os.path.isdir(outPathY):
        os.makedirs(outPathY)

    restart = r'/home/mbarlage/data/output/analysis/RESTART.'+str(start)+'_DOMAIN1'
    restart_output = outPathY+'/RESTART.'+ str(start)+'_DOMAIN1'
    if not os.path.exists(restart_output):
        cmd = "ncks -d south_north," + str(int(y)) + "," + str(int(y)) + " -d west_east," + str(int(x)) + "," + str(int(x)) + " " + restart + " " + restart_output
        os.system(cmd)

    paramList = GetFiles(NWMPath, outPathY, int(start), int(end),irri_record)
    #paramList = filter(check, paramList)
    process_bar = ShowProcess(len(paramList), 'OK')
    #print(len(paramList))
    for p in paramList:
        if os.path.exists(p[1]) and p[3]==False:
            process_bar.show_process()
            continue
        if os.path.exists(p[1]) and p[3] == True:
            os.system("rm " + p[1])
        CreateLDASIN(p[0],p[1],x,y,p[2])
        process_bar.show_process()



def main():
    site = 'UNL_KELLY_2020'
    x = 986
    y = 448
    start = '2020050100'
    end = '2020103123'
    irri_record = []
    para_kelly2020 = {'lon': -98.2034475, 'lat': 41.9480136, 'wp': 0, 'awc': 0, 'planting_date': '2020050100',
                      'target_date': '2020103123', 'irr_record': [],
                      'x': 986, 'y': 448, 'site_name': "UNL_KELLY_2020", 'dep_thld': [22, 35, 52, 15], 'ETorSM': "SM",
                      'crop_type': "Corn"}
    createForcingData(x, y, start, end, irri_record)


class ShowProcess():
    """
    show process
    """
    i = 0 # current process
    max_steps = 0 # total steps
    max_arrow = 50 #length of process bar
    infoDone = 'done'

    # initial,number of total steps is necessary
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # show function, show process according to current process
    # looks like[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #count '>'
        num_line = self.max_arrow - num_arrow #count'-'
        percent = self.i * 100.0 / self.max_steps #calculate process, in format xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #string with output, '\r'back to left without wrap
        sys.stdout.write(process_bar) #print to terminal
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

if __name__ == "__main__":
    #AdjustNC('2020020119.LDASIN_DOMAIN1','2020020119.LDASIN_DOMAIN1')
    main()
    #test()
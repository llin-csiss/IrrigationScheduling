import glob, os
def deleteDays(x,y,irri_record):
	dirname = "/home/hzhao/single-point/analysis/"+str(int(x))+"/"+str(int(y))+"/"
	if len(irri_record) > 0:
		for item in irri_record:
			date = item["start"].split('T')[0].replace('-', '')
			filename = date+"*"
			pathname = dirname + filename
			for f in glob.glob(pathname):
				os.remove(f)
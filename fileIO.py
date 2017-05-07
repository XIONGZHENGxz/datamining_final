import csv

def writeToFile(input_path,pred):
	with open(input_path,"wb") as out_file:
		csvwriter = csv.writer(out_file,delimiter=',')
		header = ['RefId','IsBadBuy']
		csvwriter.writerow(header)
		refid = 73015
		for p in pred:
			p = int(p)
			row = [str(refid),str(p)]
			csvwriter.writerow(row)
			refid+=1

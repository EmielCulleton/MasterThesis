#import config
import pandas as pd
import json

#directly importing it as json didn't work might improve this later
f = pd.read_json("/home/emiel/data/textwash_data.json")
pandas_file = pd.DataFrame(f)


training_file_pandas  = pandas_file.fillna(method="ffill")
#print(training_file_pandas[["tokens", "spans"]].head(1))

ts = training_file_pandas[["tokens" , "spans"]].head(1)
ts_explode = ts.explode("tokens")
#print(f)

with open("/home/emiel/data/textwash_data.json", "r") as jason_f:
	label_map = json.load

print(ts)


#reading the json file for reference
json_file_open = open("/home/emiel/data/textwash_data.json")
json_file = json.load(json_file_open)
print(json_file[1])

#for token, span in ts:
#	if 

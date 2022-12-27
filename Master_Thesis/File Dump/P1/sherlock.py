import pandas as pd
import json

file = open('/home/emiel/data/textwash_data.json')
data = json.load(file)

df_to_parse = pd.read_json('/home/emiel/data/textwash_data.json')
data_file = pd.DataFrame(df_to_parse)
print(data_file['tokens'][0])


#data_string = json.dumps(data)
#print(type(data_string))




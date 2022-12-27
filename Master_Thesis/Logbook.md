##### first commit:
starting over, I think I figured out how to structure my repositories now. I have yet to figure out how to load in the .json and .xml files.

##### second commit:
BIG change in the way the code is structured. I followed a tutorial (https://www.youtube.com/watch?v=MqQ7rqRllIc \ https://github.com/abhishekkrthakur/bert-entity-extraction) in order to set up BERT for named entity recognition. However, a welcome result was that it also explained a lot on how to structure code throughout different files. This gave me a lot of insight in how to "spread out" functions and definitions and make the entirety of the project more overseeable. "Project.py" has now most likely become obsolete, but will not be deleted just yet.

I tried installing tqdm on some of the processes in order to see whether it's running and how long it takes. I also wrote some code to convert json to a csv file because I figured this was easier than finding a way to use json files as an input for BERT (for now). -- BEING LAZY DID NOT WORK -- So now I'm looking for a way to use json as an input.

##### third commit:
When parsed, json files are parsed as a python dict. Use them as such. It appears that I was making my life unnesseccarily difficult. JSON can be easily converted to a DataFrame, and the train.py file was trying to convert a DataFrame to a DataFrame. Now that the training file is succesfully found and read by the training part of the program, it is time to adapt the code to my specific task.

Currently I am looking for a way to properly structure the training file so the program can read it word for word. This way I can edit the code to fit the data.

This link will come in handy later, I think: https://skimai.com/how-to-fine-tune-bert-for-named-entity-recognition-ner/ it is handy for reporting stuff. DON'T DELETE 

#### fourth commit:
It appears I forgot to commit the third commit.

Adding new code from the HuggingFace libraries. This involved a different set of evaluation codes, I think they're similar to what I added last commit, but it looks like it gives clearer results once everything is up and running.

Not a lot ended up working. However I managed to add some code that parses through XML files and adds the contents to a list. This currently works for smaller files but gives a memory error for the COW corpus file.

Also so restructuring of the files as I have a lot of files that are currently out of use but I don't want to discard just yet.


#### fifth commit:
This commit is focused on implementing the training file into the model (https://huggingface.co/docs/transformers/tasks/token_classification#load-wnut-17-dataset)

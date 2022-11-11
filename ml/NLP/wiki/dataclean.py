import pandas
import html2text 
import time 
import sys 
import re
import os


# VARIABLES 
_ENCODING = "cp437"

# OBJECTS 
parser = html2text.HTML2Text()
parser.unicode_snob = True 


def init_data():
	if len(sys.argv) > 1:
		import gpt_2_simple as gpt
		gpt2.download_gpt2(model_name="124M")
	# READ DATA 	
	print(f"reading dataset")
	t1 = time.time()
	answers 	= pandas.read_csv("Answers.csv",encoding=_ENCODING) 
	questions 	= pandas.read_csv("Questions.csv",encoding=_ENCODING)

	# Parse data
	print(f"read dataset in {(time.time()-t1):.3f}, starting answer parsing")
	t1 = time.time()

	questions_dict = {}
	answers_dict = {}


	for a in answers.iterrows():
		line 		= a[1]
		parent_id 	= line[3]

		a_text		= re.sub(r"<(/|[a-z])*>","",line[5])
		a_text 		= f"\n\n\n\nANSWSER:|\n" + a_text + "\n|"

		if parent_id in answers_dict:
			answers_dict[parent_id] += [a_text]
		else:
			answers_dict[parent_id] = [a_text]


	print(f"parsed answers in {(time.time()-t1):.3f}, starting question parsing")
	t1 = time.time()


	for q in questions.iterrows():
		line 		= q[1]
		q_id 		= line[0]
		raw_text 	= line[5] + line[6]
		q_text		= re.sub(r"<(/|[a-z])*>","",raw_text)
		q_text		= "QUESTION: |\n" + q_text + "\n|"
		
		if q_id in answers_dict:
			for answer in answers_dict[q_id]:
				q_text += answer

		with open(rf"dataset\{q_id}.txt","w",encoding="utf_8") as file:
			file.write(q_text)
			file.close()


	print("DONE")

def concat_data():
	paths =  list(os.listdir("dataset"))
	l = int(len(paths) / 10) 

	concat = zip(paths[:l],paths[l:2*l],paths[2*l:3*l],paths[3*l:4*l],paths[4*l:5*l],paths[5*l:6*l],paths[6*l:7*l],paths[7*l:8*l],paths[8*l:9*l],paths[9*l:])

	print("writing")
	i = 0

	for q,r,s,t,u,v,w,x,y,z in concat:
		try:
			q_text = open(rf"dataset\{q}","r",encoding="utf_8").read()
			r_text = open(rf"dataset\{r}","r",encoding="utf_8").read()
			s_text = open(rf"dataset\{s}","r",encoding="utf_8").read()
			t_text = open(rf"dataset\{t}","r",encoding="utf_8").read()
			u_text = open(rf"dataset\{u}","r",encoding="utf_8").read()
			v_text = open(rf"dataset\{v}","r",encoding="utf_8").read()
			w_text = open(rf"dataset\{w}","r",encoding="utf_8").read()
			x_text = open(rf"dataset\{x}","r",encoding="utf_8").read()
			y_text = open(rf"dataset\{y}","r",encoding="utf_8").read()
			z_text = open(rf"dataset\{z}","r",encoding="utf_8").read()

			f = open(rf"data\{i}.txt","w",encoding="utf_8")

			file_text =  f"{q_text}\n\n\n\n\n{r_text}\n\n\n\n\n{s_text}\n\n\n\n\n{t_text}\n\n\n\n\n{u_text}\n\n\n\n\n{v_text}\n\n\n\n\n{w_text}\n\n\n\n\n{x_text}\n\n\n\n\n{y_text}\n\n\n\n\n{z_text}"	

			f.write(file_text)
			f.close()


			if ((i % 1000) == 0):
				print(f"{i} files writen")
		except FileNotFoundError as f:
			print("a")
		i += 1


if __name__ == "__main__":
	concat_data()

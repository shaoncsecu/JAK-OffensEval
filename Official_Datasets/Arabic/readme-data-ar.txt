========================

OffensEval 2020: Identifying and Categorizing Offensive Language in Social Media (SemEval 2020 - Task 12)
Arabic training, development, and testing data 
v 1.0: December 16 2019
https://sites.google.com/site/offensevalsharedtask/

========================

1) DESCRIPTION

The files offenseval-ar-training-v1.tsv, offenseval-ar-dev-v1.tsv, and offenseval-ar-test-v1.tsv contain 7,000, 1,000, and 2,000 annotated tweets in order. 

The file offenseval-annotation-ar.txt contains a short summary of the annotation guidelines.

Twitter user mentions were substituted by @USER and URLs have been substitute by URL.

Each instance contains up to 1 label corresponding to one of the following sub-task:

- Sub-task A: Offensive language identification; 


2) FORMAT

Instances are included in TSV format as follows:

ID	INSTANCE	SUBA

The column names in the file are the following:

id	tweet	subtask_a

The labels used in the annotation are listed below.

3) TASKS AND LABELS

(A) Sub-task A: Offensive language identification

- (NOT) Not Offensive - This post does not contain offense or profanity.
- (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

In our annotation, we label a post as offensive (OFF) if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. 

Contact

semeval-2020-task-12-all@googlegroups.com

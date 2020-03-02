========================

OffensEval 2020: Identifying and Categorizing Offensive Language in Social Media (SemEval 2020 - Task 12)
Danish Training data 
v 1.0: December 16 2019
https://sites.google.com/site/offensevalsharedtask/

========================

1) DESCRIPTION

The file offenseval-da-test-v1-nolabels.tsv contains 330 annotated social media comments. 

The file offenseval-annotation.txt contains a short summary of the annotation guidelines.

User mentions were substituted by @USER and URLs have been substituted by URL.

Each instance contains an identifier and text. The final column is left blank. 
The goal is to predict the label for the following tasks:

- Sub-task A: Offensive language identification; 


2) FORMAT

Instances are included in TSV format as follows:

ID	TEXT

The column names in the file are the following:

id	tweet	subtask_a

The labels used in the annotation are listed below.


3) TASKS AND LABELS

(A) Sub-task A: Offensive language identification

- (NOT) Not Offensive - This post does not contain offense or profanity.
- (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

In our annotation, we label a post as offensive (OFF) if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. Labels are not included in the test set.

Contact

semeval-2020-task-12-all@googlegroups.com



This data was originally created as part of:
 Offensive Language and Hate Speech Detection for Danish. Gudbjartur Ingi Sigurbergsson, Leon Derczynski (2020); Proc. LREC

If you use this dataset in your research, cite:

@inproceedings{sigurbergsson2020offensive,
    title={Offensive Language and Hate Speech Detection for Danish},
    author={Gudbjartur Ingi Sigurbergsson and Leon Derczynski},
    year={2020},
    inproceedings={Proceedings of the 12th Language Resources and Evaluation Conference},
    organization={ELRA}
 }

(used with permission)


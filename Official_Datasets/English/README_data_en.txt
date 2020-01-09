========================

OffensEval 2020: Identifying and Categorizing Offensive Language in Social Media (SemEval 2020 - Task 12)
English training data 
v 1.0: December 30, 2019
https://sites.google.com/site/offensevalsharedtask/

========================


1. DESCRIPTION

The folder contains three ZIP files, secured with a password, for each of the subtasks A, B, C, in English.
(The password is provided by the organizers after registration).

Each of the archives contains a TAB-separated file with distantly annotated instances as follows: 


1.1. FILE FORMAT FOR SUBTASKS A AND B

# task_a_distant.tsv, task_b_distant.tsv

Each line of the files for subtasks A and B is a training instance in the following format:

	ID <TAB> TWEET <TAB> AVG_CONF <TAB> CONF_STD

where:

	- AVG_CONF is the average of the confidences predicted by several supervised models for a specific instance to belong to the positive class for that subtask. The positive class is OFF for subtask A, and UNT for subtask B.

	- CONF_STD is the confidences' standard deviation from AVG_CONF for a particular instance.


1.2. FILE FORMAT FOR SUBTASK C

# task_c_distant.tsv

Each line of the file for subtask C is a training instance given in the following format:

	ID <TAB> TWEET <TAB> AVG_CONF_IND <TAB> AVG_CONF_GRP <TAB> AVG_CONF_OTH <TAB> CONF_STD_IND <TAB> CONF_STD_GRP <TAB> CONF_STD_OTH

where:
	- AVG_CONF_<CLS> are the averages of the confidences predicted by several supervised models for a specific instance to belong to the corresponding class <CLS>: IND, GRP, OTH.

	- CONF_STD_<CLS> is the confidences' standard deviation for a particular instance and a corresponding class.


2. TASKS AND LABELS

* (A) Subtask A: Offensive Language Identification

	- (NOT) Not Offensive: A post with no offensive language or profanity.
	- (OFF) Offensive: A post containing offensive language or a targeted (veiled or direct) offense.

	In our annotation, we label a post as offensive (OFF) if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. 

* (B) Level B: Automatic Categorization of Offense Types

	- (TIN) Targeted Insult and Threats: A post containing an insult or a threat to an individual, a group, or others (see categories in sub-task C).
	- (UNT) Untargeted: A post containing non-targeted profanity and swearing.

	Posts containing general profanity are not targeted, but they contain non-acceptable language.

* (C) Level C: Offense Target Identification

	- (IND) Individual: The target of the offensive post is an individual: a famous person, a named individual, or an unnamed person interacting in the conversation.
	- (GRP) Group: The target of the offensive post is a group of people considered as a unity due to the same ethnicity, gender or sexual orientation, political affiliation, religious belief, or something else.
	- (OTH) Other: The target of the offensive post does not belong to any of the previous two categories (e.g., an organization, a situation, an event, or an issue).


* Label Combinations

Here are the possible label combinations in the OLID annotation:

	- NOT NULL NULL
	- OFF UNT NULL
	- OFF TIN (IND|GRP|OTH)



TASK ORGANIZERS

Marcos Zampieri - Rochester Institute of Technology, USA
Preslav Nakov - Qatar Computing Research Institute, Qatar
Sara Rosenthal - IBM Research, USA
Pepa Gencheva - University of Copenhagen, Denmark
Georgi Karadzhov - University of Cambridge, UK

(for non-English languages; not included here)
Hamdy Mubarak - Qatar Computing Research Institute, Qatar
Leon Derczynski - IT University Copenhagen, Denmark
Zeses Pitenis - University of Wolverhampton, UK
Çağrı Çöltekin - University of Tübingen, Germany


TASK WEBSITE

	https://sites.google.com/site/offensevalsharedtask/


TASK CONTACT

	semeval-2020-task-12-all@googlegroups.com

------

HOW TO CITE THIS DATASET:

@inproceedings{zampieri-etal-2020-semeval,
    title = "{S}em{E}val-2020 Task 12: Multilingual Offensive Language Identification in Social Media ({O}ffens{E}val 2020)",
    author = "Zampieri, Marcos  and
      Nakov, Preslav  and
      Rosenthal, Sara  and
      Atanasova, Pepa and
      Karadzhov, Georgi and
      Mubarak, Hamdy and
      Derczynski, Leon and
      Pitenis, Zenes and
      Çöltekin, Çağrı",
    booktitle = "Proceedings of the 14th International Workshop on Semantic Evaluation",
    series = "SemEval~'20"
    month = jun,
    year = "2020",
    address = "Barcelona, Spain",
}

------

REFERENCES

	Basile, V., Bosco, C., Fersini, E., Nozza, D., Patti, V., Pardo, F.M.R., Rosso, P. and Sanguinetti, M., (2019) Semeval-2019 task 5: Multilingual detection of hate speech against immigrants and women in twitter. In Proceedings of the 13th International Workshop on Semantic Evaluation (pp. 54-63).

	Davidson, T., Warmsley, D., Macy, M. and Weber, I. (2017) Automated Hate Speech Detection and the Problem of Offensive Language. Proceedings of ICWSM.

	Kumar, R., Ojha, A.K., Malmasi, S. and Zampieri, M. (2018) Benchmarking Aggression Identification in Social Media. In Proceedings of the First Workshop on Trolling, Aggression and Cyberbullying (TRAC). pp. 1-11.

	Malmasi, S., Zampieri, M. (2018) Challenges in Discriminating Profanity from Hate Speech. Journal of Experimental & Theoretical Artificial Intelligence. Volume 30, Issue 2, pp. 187-202. Taylor & Francis. 

	Waseem, Z., Davidson, T., Warmsley, D. and Weber, I. (2017) Understanding Abuse: A Typology of Abusive Language Detection Subtasks. Proceedings of the Abusive Language Online Workshop.


PREVIOS OffensEval

	REPORT
		Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N. and Kumar, R. (2019) SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval). In Proceedings of the 13th International Workshop on Semantic Evaluation. pp. 75-86.

	DATASET
		Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N. and Kumar, R. (2019) Predicting the Type and Target of Offensive Posts in Social Media. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). pp. 1415-1420.

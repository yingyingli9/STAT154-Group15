# MAG-A
Stat 154 Final Project
Classification of Hillary Email!! (Topic modelling——hard,forget it==)
• Training Set 3000 email, Label sender. Word feature matrix. [Classifying the sender]
Tips:
  • Word feature matrix: for each column, some words counts. (how to choose the words matters!)
	• Treat the whole email as string at beginning, find the lowest or strange words, find his own special structure of using words.
	• Turn into lower case, remove the stop words, do stamping (classifying the same words, tense adjustment) . NLTK in python package
	• Loop through email, create a dictionary of Dict={"doe":[0,2,0,…]} (each words have a key, each key contains the minimize the letter the words have). (confusing O_0 R&F, SPIAN?)
	• Classified the standard form of world feature matrix input and output form.
	• Do your classification unsupervised using k-mean.
	• Finding power feature. In random forest. 
	• Run your project, run a test set and try to get better fit than the training test. 


Dashboard produced by Artemis Mohseni, Spring 2020
---------------------------------------

# Short summary about Dataset
Imbalanced dataset typically refers to classification problems where the classes are not represented equally. In this dashboard, an imbalanced dataset contained more than 2.3 million samples including the learning behaviors of 6,423 students who used an online study tool with 1,445 educational topics for the period of one year is used. Each educational topic and every student have a unique ID. Each sample of the dataset describes some information including student ID, topic ID, question type, resource name, resource type, student answer, answer duration and the month, the day and the hour that each student answers a question. 
In this tool, the students chose their own path through the material, which is arranged in books and chapters. The activity of each student in a topic reflects how many questions she/he answered to that topic. Because of True/False type questions, all students' answers should have been either true or false. However, many students started a quiz but did not answer all the questions, resulting in many Null values in the dataset which cause outliers. In order to solve the mentioned problem and propose a good visualization, cleaning and preprocessing data is really important. Python, Jupyter notebook (Project Jupyter, 2020) is used to clean the imbalanced dataset and preprocess data. By applying this step, the number of samples is reduced to 1,322,097. In order to decrease the loading time in the dashboard, 10,000 random samples were selected which still allow us to answer all the below research questions. 
Since half of the students used the online tool just one time and they stopped using the tool, therefore, they didn't register their university IDs in the tool. As a result, in the dataset, there are university IDs for the students who used the tool, regularly.  In order to analyze the dataset, we assigned a unique ID to all the students who didn't have a university ID. Since the number of answers for each topic ID should not be less than 100, which is too low for a meaningful analysis, so a filter is applied for the topic IDs with a low population.
The student answer duration in the dataset is between 0 and 25941 minutes (432 hours). This value shows that a large number of students start to answer a question and they have never finished it. Therefore, there are much unusable data that should be dropped. A filter for answer duration is applied which considers the duration between 0 and 20 minutes.

# Goal of Developing a Dashboard
The focus of this research is to develop a dashboard to interact with and analyze the data, deliver the most relevant information to the teachers and help them to understand the data, and make faster and accurate decisions. Also, defining research questions is an important step to find the road map and speed up the process of creating a dashboard.  Research questions are as follow:

RQ1.	How many correct and incorrect answers are there in general? 
RQ2.	What are the maximum and minimum time students spend to answer a question?
RQ3.	How many correct and incorrect answers are there for each student? 
RQ4.	How many correct and incorrect answers are there for every topic? 
RQ5.	How many correct and incorrect answers are there per day? 
RQ6.	How many correct and incorrect answers are there per hour? 
RQ7.	What are the percentages of correct and incorrect answers in general?
RQ8.	How many correct and incorrect answers are there in different time categories? 
RQ9.	How many correct and incorrect answers are there in each student category?
RQ10.	How many correct and incorrect answers are there in each topic category?
RQ11.	How many correct and incorrect answers are there in each question type?
RQ12.	How many correct and incorrect answers are there in each month of year/day of month/hour of day?
RQ13.	What hour of day students are more active?
RQ14.	What day of month students are more active?
RQ15.	What month of year students are more active?
RQ16.	Which topic has the highest number of correct answers? 
RQ17.	What are the top 10 topics that students are more interested? 
RQ18.	What is the number of correct answers for top 10 topics?
RQ19.	Which question type is more interesting for students to answer it? 
RQ20.	What is the percentage of students’ activities per month?


# Proposed Dashboard
A dashboard which provides a central location to monitor and analyze the data is the most efficient way to track multiple data sources. Real-time monitoring reduces the hours of analyzing and long line of communication. 
The main aim of the proposed dashboard is to provide an interactive platform for teachers so they can compare and explore students’ performance by considering different features such as answer duration, topic IDs, student IDs, question type, monthly, daily and hourly students’ activities. This dashboard also allows to pick two special categories belong to each mentioned feature at a time and compare them by different features. Some interactive techniques such as brushing, zoom, pan and filter are supported in the dashboard. In brushing technique, a subset of elements is selected in one view and this selection is updated and maintained in other views.

![GitHub Logo](/Dashboard_Month.png)

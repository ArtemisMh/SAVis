Dashboard by Artemis, Spring 2020
---------------------------------------

Short summary In this dashboard an imbalanced dataset including the learning behaviors of 6,423 students who used an online study tool with different educational topics for the period of one year is used. In this work, after preprocessing and cleaning data, we created a csv file with 1,322,097 samples including 6,423 learning vectors (one per student) in the dataset, each with 1,445 topics. In order to decrease the loading time in the dashboard, 10,000 samples are selected randomly. Since lots of students didn't use the online tool in the appropriate way, there are many unusable data. Proposing a good plot without doing data filtering in uncleaned dataset is not achievable. As a result, preprocessing data and data cleaning are really important. Preprocessing and cleaning data The topic ID in this dataset starts from 665. Also, for the topic IDs from 3600-6182 the number of correct and incorrect answers are not impressive. Therefore, a filtering for the topic IDs between 650 and 3500 is applied. The student answer duration in the dataset is between 0 and 25941 minutes (432 hours). This value shows that lots of students start to answering a question and they never finished it. As a result, there are lots of unusable data that should be dropped. So, a filtering for the student answer time between 0 and 20 minutes is considered. Goal of developing a dashboard The focus of this research is to create a dashboard to interact with and analyze the data, deliver the most relevant information to the teachers and help them to understand the data, and make faster and accurate decisions. Before developing a dashboard, we should clean, preprocess and then analyze the data. Also, defining research questions is an important step to find the road map and speed up the process of creating a dashboard. In order to analyze the data T-test is used. T-test which usually comes from histogram is a type of inferential statistic used to determine if there is a significant difference between the means of two groups of numerical or quantitative data, which may be related in certain features. this test is not good for big data. In order to calculate the P-value in T-test which is achievable for N<5000, 100 random samples are chosen. P-value for T-test should be less than alpha level to reject the null hypothesis (H0). Research questions and a null hypothesis are as follow: 


RQ1. How many correct/incorrect answers are there in general? 
RQ2. How many correct/incorrect answers are there in different time categories? 
RQ3. How many correct/incorrect answers are there in each student category? 
RQ4. How many correct/incorrect answers are there in each topic category? 
RQ5. How many correct/incorrect answers are there in each question type? 
RQ6. How many correct/incorrect answers are there in each month of year/day of month/hour of day? 
RQ7. Which topic has the highest correct answers? 
RQ8. What are the top 10 topics that students are more interested? 
RQ9. Which question type is more interesting for students to answer it?
RQ10. What are the number of correct answers for top 10 topics? 
RQ11. What is the maximum/minimum time students spend to answer a question? 
RQ12. What hour of day students are more active? 
RQ13. What day of month students are more active? 
RQ14. What month of year students are more active?

H0. Expected value in null hypothesis for average answer duration is minute. 


Proposed Dashboard A dashboard which provides a central location to monitor and analyze the data is the most efficient way to track multiple data sources. Real-time monitoring reduces the hours of analyzing and long line of communication. The main aim of the proposed dashboard is to provide an interactive platform for teachers so they can compare and explore students’ performance by considering different features such as answer duration, topic IDs, student IDs, question type, monthly, daily and hourly activities. This dashboard also allows to pick two special categories belong to each mentioned feature at a time and compare them by different features. Some interactive techniques such as brushing, zoom, pan and filter are supported in the dashboard. In brushing technique, a subset of elements is selected in one view and this selection is updated and maintained in other views. 

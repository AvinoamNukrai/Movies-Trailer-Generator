# Movies-Trailer-Generator
In this project, I developed an automated system for creating movie trailers. The project comprises several building blocks:

![image](https://github.com/user-attachments/assets/30654fb8-1ad4-4445-a688-eeebc53d675c)


The most important models of the project are the "spoiler detector" and "scene to frame" models. 

The purpose of these models is to identify if some scene represented by text contains spoilers or not, given the movie plot, and to identify the exact frame where the scene occurs in the film, respectively.

To address these challenges, I utilized machine learning models such as Logistic Regression and Random Forest, among others. Additionally, I used OpenAI models like GPT and CLIP.

The "Spoiler Detector" model implementation details: 

![image](https://github.com/user-attachments/assets/9066453e-e783-4e90-96b0-605ede1ebfdd)


![image](https://github.com/user-attachments/assets/db190752-34ce-4803-845a-a5c382034d40)


![image](https://github.com/user-attachments/assets/c17e7a1c-8ce9-4e5e-8d7b-81e761810646)


![image](https://github.com/user-attachments/assets/c968d0cf-dcb3-417e-85c5-c3b8c41a4c5e)


![image](https://github.com/user-attachments/assets/83639a3d-344f-4dcd-aef8-f724d6338dca)


The "Scene To Frame" model core algorithm:  

![image](https://github.com/user-attachments/assets/f04822e9-44b3-4b6d-b063-d8f0a5f0e350)


For more details about the project take a look on the next 2 presentations (Spoiler Detector & Trailer Generator): 


https://www.canva.com/design/DAGLjmSK4eg/RnYIv6d_0thE8NjYvw9BWQ/edit?utm_content=DAGLjmSK4eg&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


https://www.canva.com/design/DAGI3bPGux8/n9JeCmd0jbHK0biFAehtjg/edit?utm_content=DAGI3bPGux8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Example: 

https://github.com/user-attachments/assets/4fee93a7-6a78-4331-8fe9-65331729504c


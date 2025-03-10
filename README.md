# DL_Assign1

### Github link
https://github.com/Kushshahch21b053/DL_Assign1


### Report link
https://wandb.ai/ch21b053-indian-institute-of-technology-madras/DL_Assign1/reports/CH21B053-A1-Report--VmlldzoxMTU0NDkyMA


### Code organisation

src
- dataset.py
- model.py
- backprop.py
- optimizers.py

train.py

best_model.py

mnist_q10.py

### How to train and evaluate

 - Question 1
   To generate sample images, we have to run python src/dataset.py if in the root directory

- Questions 4,5,6
  To make the sweep and evaluate the models, the following line should be run in the command line
  For questions 4, 5 and 6, the code can be run using python train.py --wandb_entity myname --wandb_project myprojectname     in the command line while in the root directory

- Question 7
  python best_model.py --loss_type cross_entropy     in the command line while in the root directory

- Question 8
  python best_model.py --loss_type mse      in the command line while in the root directory

- Question 10
  pyhton mnist_q10.py     in the command line while in the root directory
  

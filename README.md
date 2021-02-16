# Automatic Generation of Rhetorical Questions and Its Application to a Chatbot

Clone this project:
```script
github clone https://github.com/sun510001/RQ_Chatbot.git
```

## Introduction
In recent years, an interpersonal attraction in the conversation has been studied extensively.
Creating figurative language generation modules can make chatbots more human-like.
Some recent studies have proposed chatbots that generate sarcasm. 
However, they do not focus on generating rhetorical questions (RQ).
It is necessary for chatbots to generate RQs to be more human-like because RQs are usually used in daily conversation 
and social media dialog.
RQs are questions but not meant to obtain an answer.
People usually use them to express their opinions in conversation. However, a question cannot be recognized as an RQ if 
the answer of the question is only known by the speaker. 
To recognize that it is an RQ, the listener needs to use the knowledge shared between them. 
Furthermore, there is a specific interrelation between irony and RQs. Therefore RQs are always used to express their 
negative opinions.
Questions based on the valence-reversed commonsense knowledge can be easily recognized as RQs because both speaker and 
listener know their answers are negative.
For example, the commonsense knowledge ``Giving money to the poor will make good world`` can be converted into an RQ: ``
Will giving money to the rich make a good world?``

This study aims to generate a negative-answering RQ by using valence-reversed commonsense knowledge sentences to make 
the chatbot more appropriate and human-like in a conversation. Additionally, we use a situation classifier analyzing 
previous contexts to decide when to generate a literal response, sarcastic response, and RQ.

You can get more information by reading 
the [final_report_fin_version.pdf](https://github.com/sun510001/RQ_Chatbot/blob/master/final_report_fin_version.pdf).

## Preparations
1. Install the environment
    ```script
    conda env create -f environment.yml
    ```
2. Download models
   * Transformers pre-training models
        ```script
        cd Situation_Classification_for_SRL/
        python run_preproc.py
        ```
   * Fine-tuned models
       ```script
       Download files from https://drive.google.com/drive/folders/1XlXAV2fIEeTSwyBMx0dKCVevsA3XfWM_?usp=sharing
       cd Master_research_model
       mv roberta-base_model Situation_Classification_for_SRL/data/
       mv bert-base-uncased-model RQ_generator/data/
       ```
3. Download the sarcasm generation module, set the module by reading it's README.md and then replace 
   files.
    ```script
    clone https://github.com/tuhinjubcse/SarcasmGeneration-ACL2020.git
    cd SarcasmGeneration-ACL2020/
    cat README.md
    do settings... 
    mv ../sg_file/* SarcasmGeneration-ACL2020/
    ```
4. Setting for RQ generator module.
    * Download bert-gec
        ```script
        cd RQ_generator/
        git clone https://github.com/kanekomasahiro/bert-gec.git
        ```
    * Commonsense knowledge representation model for scoring arbitrary tuples.
        ```script
        cd RQ_generator/data
        wget https://ttic.uchicago.edu/~kgimpel/comsense_resources/ckbc-demo.tar.gz
        tar -xvzf ckbc-demo.tar.gz
        rm ckbc-demo.tar.gz
        ```
    * Download stanford-parser-4.2.0.zip 
        ```script
         wget https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip
         tar -xvzf stanford-parser-4.2.0.zip
         rm stanford-parser-4.2.0.zip
        ```
## The Chatbot Mode
You can run the run_chatbot.py directly after you did [preparations](#Preparations).
```script
Python run_chatbot.py
```

## The Evaluation Mode
The evaluation mode can output all types of responses in any situation of a conversation.
1. Generate literal responses
    * Uncomment codes that are under the ``predict/for evaluation`` in run_predict.py, 
       and comment out all codes that are under the ``for chatbot``.
    * Run the literal generator
        ```script
        python run_generate_evaluation.py
        ```
  
2. Generate the situation classification
    ```script
    cd Situation_Classification_for_SRL/
    python run_predict.py
    ```

3. Generate the sarcastic responses
    ```script
    clone https://github.com/tuhinjubcse/SarcasmGeneration-ACL2020.git
    mv generate_sarcasm.py SarcasmGeneration-ACL2020/
    ```

    * Change the conda_path to your python environment path
        ```script
        cd SarcasmGeneration-ACL2020/
        vim generate_sarcasm.py
        conda_path = '/home/aquamarine/sunqifan/anaconda3/envs/r_cla/bin/python3.6'
        python generate_sarcasm.py
        ```

4. Generate the RQ responses
    ```script
    python run_train_classifier.py
    ```
    * If your memory or GPU memory is not enough for running whole data in the dataset, you can run it in a section.
    

## Training models
If you want to train models for the situation classification and the RQ generator, please read it.
### Situation classification for SRL (Sarcastic, Rhetorical question and Literal responses)
* We use the dataset
from [Twitter and Reddit data for the Shared Task](https://github.com/EducationalTestingService/sarcasm)
* Pre-processed dataset is ``sarcasm_merge_triple_v8.csv`` in ``Situation_Classification_for_SRL/data/``.
* You can set the type of training models in ``__init__.py/TrainModelConfig``.
    ```script
    cd Situation_Classification_for_SRL/
    python run_train.py
    ```

### RQ_detection in RQ generator
* You can set the type of training models in ``__init__.py/TrainModelConfigV2``.
    ```script
    cd RQ_generator/
    python run_train.py
    ```

## Citing Us
Please email me at sqf121@gmail.com for any problems/doubts. Further, you can raise issues on Github or suggest improvements.
Please leave a star and cite us if you use our code, data, or report
```script
@article{RQChatbot,
author = {Qifan Sun},
title = {Automatic Generation of Rhetorical Questions and Its Application to a Chatbot},
year = {2021}
}
```

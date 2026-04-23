Hi, my name is Mykyta

This is my keyword detection project. My goal is to achieve a high accuracy in recognizing 7 keywords
and then implement this model into a demo game about a dog. Data was collected from Google Speech Commands,
noise audio data (Kaggle), and my own samples from another project. 

Classes:
0 - noise
1 - marvin
2 - go
3 - stop
4 - left
5 - right
6 - yes
7 - no

keyword_v2.tflite is a CNN with dilated blocks trained on roughly 46 samples. Testing results are as following:

     precision    recall  f1-score   support

       noise       0.83      1.00      0.90      8966
      marvin       0.99      0.87      0.92      2794
          go       0.95      0.82      0.88      3795
        stop       0.99      0.93      0.96      3808
        left       0.92      0.90      0.91      3765
       right       0.95      0.91      0.93      3787
         yes       0.97      0.92      0.95      3803
          no       0.90      0.84      0.87      3800

    accuracy                           0.91     34518
   macro avg       0.94      0.90      0.92     34518
weighted avg       0.92      0.91      0.91     34518

To run the app you have to enter the 
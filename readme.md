#Potato Disease Classification task
This is a system which takes an uploaded image of a potato plant and using the trained model 
classifies it as one of the 3 classes of diseases. This project I believe has agricultural benefits and helps
farmers ideentify early signs of disease and prvent than as early as possible. Its development is based entirely on 
CNNs. The resulting model has above 92 percent accuracy.

How to run:

  uvicorn main:app --reload

Properties:

  -- Data preprocessing and augmentation to prevent overfitting
  -- CNN with multiple layers of convolution, pooling, activation following each other, and one fully connected layer in the end
  -- Compiled using Adam optimizer

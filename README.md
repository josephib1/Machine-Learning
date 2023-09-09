# Machine-Learning

Project Setup:
To run this project, you need to set up the following:

1.	Python: Ensure that you have Python installed on your system. You can download the latest version of Python from the official Python website ( https://www.python.org ) and follow the installation instructions for your operating system.

2.	TensorFlow: Install the TensorFlow library, which is used for machine learning and neural network operations. Open a command prompt or terminal and run the following command to install TensorFlow:
pip install tensorflow

3.	NumPy: Install the NumPy library, which is required for numerical operations and array manipulation. Run the following command in your command prompt or terminal:
pip install numpy

4.	Tkinter: Tkinter is a standard Python library for creating GUI applications. It is usually included with Python installations. However, if you don't have Tkinter installed, you may need to install it separately based on your operating system.
For Windows: Tkinter is typically included with the standard Python installation on Windows.

Once you have set up Python, installed the required libraries (TensorFlow, NumPy, and Tkinter), you should be ready to run the project. Execute the Python script containing the code, and the graphical interface for character recognition will be displayed.

Introduction:

The aim of this project is to develop a program with a graphical interface that can recognize lowercase and uppercase alphabetic characters. The program utilizes machine learning techniques, specifically a neural network, to achieve this task. The project can be divided into several key steps.

Step 1: Creating a Learning Base:
To train the neural network, a learning base is generated. This learning base contains alphabetic characters, both lowercase and uppercase. Additionally, variations of these characters are included to enhance the diversity of the dataset.

Step 2: Designing a Neural Network Structure:
A suitable neural network structure is designed for the character recognition task. The chosen architecture consists of a sequential model with multiple dense layers. The input layer size is determined as 152, while the output layer size is set to 2, representing the two classes: lowercase and uppercase.

Step 3: Applying the Gradient Backpropagation Algorithm:
The gradient backpropagation algorithm is applied to train the neural network. The model is compiled using the Adam optimizer and categorical cross-entropy loss function. This enables the network to learn from the training data and improve its performance.

The Code:
The provided code implements the aforementioned steps. It first generates the learning base, preprocesses the data, and trains the neural network using the backpropagation algorithm. The resulting model is then used to test the recognition of characters.
Note: All the functions used are commented in the code.

The program also includes a graphical user interface (GUI) developed using the Tkinter library. The GUI allows the user to select a text file containing characters to be recognized. The selected characters are extracted, preprocessed, and passed through the trained model for prediction. The results are displayed in a separate window and can also be saved to an output file.

key features in the provided code:

1.	Dataset Generation: The generate_dataset() function creates a diverse learning base by including lowercase and uppercase alphabetic characters along with additional variations. This ensures a rich and representative dataset for training the model.
 num_variations = 100
increment this line of code to get more variations of lowercase and uppercase characters.
Don’t forget to increment also input_size(line 46) to the value=num_variations+52 where 52 is the number of upper and lower case characters.

2.	Neural Network Architecture: The neural network is designed using the Keras Sequential API. The model consists of multiple dense layers, with the activation function set to ReLU for intermediate layers and softmax for the output layer. This architecture allows the network to learn complex patterns and make accurate predictions.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])


3.	Model Training: The train_model() function trains the neural network using the compiled model and the preprocessed training data. It uses the fit() function and specifies the number of epochs and batch size. Increasing the number of epochs can improve the convergence of the model.
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32)  # Increase epochs for better convergence


4.	Data Preprocessing: The preprocess_data() function converts the dataset into a suitable format for training the neural network. It maps characters to their corresponding indices and creates one-hot encoded vectors as input features. The labels are also encoded as categorical variables.
def preprocess_data(data):
    char_to_idx = {char: idx for idx, char in          enumerate(string.ascii_lowercase + string.ascii_uppercase)}
    num_samples = len(data)
    X = np.zeros((num_samples, input_size))
    y = np.zeros((num_samples, output_size))

    for i, (char, label) in enumerate(data):
        X[i, char_to_idx[char]] = 1.0
        if label == "lowercase":
            y[i, 0] = 1.0
        elif label == "uppercase":
            y[i, 1] = 1.0

    return X, y


5.	GUI Integration: The code integrates a graphical user interface (GUI) using the Tkinter library. It allows the user to select a text file containing characters for recognition. The selected characters are then extracted and passed through the trained model.

6.	Result Display and Saving: The program displays the recognition results in a separate window using a scrolled text widget. The results include the character and its predicted label. Additionally, the program provides the option to save the results to an output file for further analysis.

Some screenshots of the project:


First screen prompt us to select a file that contains a text:

![test](https://github.com/josephib1/Machine-Learning/assets/105210115/41653177-5784-4a8f-972d-495606505640)
 
Next, when we choose a file, the code runs and returns the output in a screen and create a new output.txt and save the output in it:

![test1](https://github.com/josephib1/Machine-Learning/assets/105210115/9e14692f-c3dc-45fb-b6e0-0d21dd1feb68)

This is the output.txt file content:

![test2](https://github.com/josephib1/Machine-Learning/assets/105210115/10026367-391f-4d97-8725-d34c08af2069)


 


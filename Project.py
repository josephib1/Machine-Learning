import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import tensorflow as tf
import string


# Step 1: Create a learning base containing alphabetic characters
def generate_dataset():
    dataset = []
    lowercase_chars = string.ascii_lowercase
    uppercase_chars = string.ascii_uppercase

    for char in lowercase_chars:
        dataset.append((char, "lowercase"))

    for char in uppercase_chars:
        dataset.append((char, "uppercase"))

    # Additional variations
    variations = []
    num_variations = 100#edit this to change the variations num and edit input_size accordngly 
    for _ in range(num_variations):
        lowercase_char = random.choice(lowercase_chars)
        uppercase_char = random.choice(uppercase_chars)

        # Add lowercase variation
        lowercase_variation = random.choice(lowercase_chars)
        while lowercase_variation == lowercase_char:
            lowercase_variation = random.choice(lowercase_chars)
        variations.append((lowercase_variation, "lowercase"))

        # Add uppercase variation
        uppercase_variation = random.choice(uppercase_chars)
        while uppercase_variation == uppercase_char:
            uppercase_variation = random.choice(uppercase_chars)
        variations.append((uppercase_variation, "uppercase"))

    dataset.extend(variations)

    return dataset


# Step 2: Design a suitable neural network structure
input_size = 152
output_size = 2

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

# Step 3: Apply the gradient backpropagation algorithm for learning
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Load and preprocess the data
def preprocess_data(data):
    char_to_idx = {char: idx for idx, char in enumerate(string.ascii_lowercase + string.ascii_uppercase)}
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


# Train the model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10, batch_size=32)  # Increase epochs for better convergence


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


# Test the model
def test_model(model, X_test, chars):
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    mapping = {0: "lowercase", 1: "uppercase"}
    results = []
    for i, predicted_label in enumerate(predicted_labels):
        results.append(f"Character: {chars[i]}, Predicted Label: {mapping[predicted_label]}")

    return results


# Generate the dataset
train_data = generate_dataset()

# Load and preprocess the data
X_train, y_train = preprocess_data(train_data)

# Train the model
train_model(model, X_train, y_train)


# Create the Tkinter application window
window = tk.Tk()
window.title("Character Recognition")
window.geometry("500x400")


# Define the event handler for the "Check Characters" button
def check_characters():
    # Open file dialog to select the input file
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        # Read the contents of the file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()


        # Extract the characters from the content
        test_chars = [char for char in content 
                      if char.isalpha()
                      ]

        if len(test_chars) > 0:
            # Preprocess the data
            X_test, _ = preprocess_data([(char, "") for char in test_chars])

            # Test the model
            results = test_model(model, X_test, test_chars)

            # Create a scrolled text widget to display the results
            result_window = tk.Toplevel(window)
            result_window.title("Character Recognition Results")
            result_window.geometry("400x300")

            scrollbar = ttk.Scrollbar(result_window)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            result_text = tk.Text(result_window, wrap=tk.WORD, yscrollcommand=scrollbar.set)
            result_text.pack(fill=tk.BOTH, expand=True)

            scrollbar.config(command=result_text.yview)

            for result in results:
                result_text.insert(tk.END, result + "\n")

            # Write the results to a file
            output_file = "output.txt"
            with open(output_file, "w") as file:
                file.write("\n".join(results))

            messagebox.showinfo("Output Saved", f"The output has been saved to {output_file}")

        else:
            messagebox.showwarning("No Characters Found", "The selected file does not contain any alphabetic characters.")


# Create the "Check Characters" button
check_button = tk.Button(window, text="Insert Characters File", command=check_characters)
check_button.pack()

# Start the Tkinter event loop
window.mainloop()

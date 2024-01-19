import random

def generate_password(length):
    # Define characters string to use in the password
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:'\",.<>/?`~"

    # Generate the password using random.choice
    password = ''.join(random.choice(characters) for i in range(length))
    
    return password


# Get user input for the desired length of the password
length = int(input("Enter the length of the password: "))

# Generate and display the password
password = generate_password(length)
print("Generated Password:", password)

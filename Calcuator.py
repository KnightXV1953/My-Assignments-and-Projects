import numpy as np

def sum(a, b):
    return a + b

def minus(a, b):
    return a - b

def divide(a, b):
    return a / b

def multiply(a, b):
    return a * b

while True:
    print("================ ** Welcome To My Calculator** =========================")
    print("Enter operation you want to perform? +, /, -, *")
    print("Press + for sum.")
    print("Press - for subtraction.")
    print("Press / for division.")
    print("Press * for multiplication.")
    
    operation = input()

    if operation not in ['+', '-', '*', '/']:
        print("Invalid operation. Exiting.")
        break

    print("Enter two numbers you want to apply operations on:")
    a = int(input())
    b = int(input())

    if operation == "+":
        result = sum(a, b)
    elif operation == "-":
        result = minus(a, b)
    elif operation == "/":
        result = divide(a, b)
    elif operation == "*":
        result = multiply(a, b)

    print("The result is", result)

    print("Do you want to perform another operation? (yes/no)")
    user_input = input().lower()

    if user_input != 'yes':
        print("Exiting calculator. Goodbye!")
        break
